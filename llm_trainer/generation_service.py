from typing import Optional, List, Mapping, Tuple, Protocol, Dict, Any
import gc
import concurrent.futures
import torch
import torch.distributed as dist
from llm_model import (
    LlmModel,
    ModelConfig
)

from .tools import TrainerTools
from .train_configs import GenerateConfig
from .utils import get_dtype, empty_cache
from .generate_utils import (
    batch_generate,
    generate
)
from .partition_utils import (
    get_full_state_dict_on_rank0,
    unwrap_model,
    unwrap_model_for_generation
)


class GenerationServiceBase:
    def _get_clean_state_dict_on_rank0(self, model: torch.nn.Module):
        state_dict = get_full_state_dict_on_rank0(model)
        if TrainerTools().parallel.is_main_process:
            unwrapped_model = unwrap_model(model)
            prefix = 'policy_model.' if hasattr(unwrapped_model, 'policy_model') else ''
            clean_state_dict = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

            del state_dict
            return clean_state_dict

        return None


class SyncCentralGenerationService(GenerationServiceBase):
    def __init__(
            self,
            model_config: ModelConfig,
            generation_device: str,
            chunk_size: int = -1
    ):
        self.generation_device = generation_device
        self.chunk_size = chunk_size
        self.world_size = TrainerTools().parallel.world_size
        self.rank = TrainerTools().parallel.global_rank

        if TrainerTools().parallel.is_main_process:
            self.gen_model = LlmModel(model_config).to(self.generation_device, get_dtype())
            self.gen_model.eval()
            for param in self.gen_model.parameters():
                param.requires_grad = False
        else:
            self.gen_model = None

    def __call__(
            self,
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            generate_config: GenerateConfig,
            task_type: str,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ) -> List[List[int]]:

        state_dict = self._get_clean_state_dict_on_rank0(model)
        if TrainerTools().parallel.is_main_process:
            self.gen_model.load_state_dict(state_dict)

            del state_dict
            gc.collect()
            empty_cache()

        local_prompts = prompt_ids.cpu().tolist()
        gathered_prompts = [None for _ in range(self.world_size)]

        if self.world_size > 1:
            dist.all_gather_object(gathered_prompts, local_prompts)
        else:
            gathered_prompts[0] = local_prompts

        gathered_results = [[] for _ in range(self.world_size)]

        if TrainerTools().parallel.is_main_process:
            for target_rank in range(self.world_size):
                target_prompts = gathered_prompts[target_rank]

                if task_type == 'eval' and target_rank != 0:
                    gathered_results[target_rank] = []
                    continue

                if not target_prompts:
                    gathered_results[target_rank] = []
                    continue

                target_prompts_tensor = torch.tensor(target_prompts, dtype=torch.long, device=self.generation_device)
                attention_mask = (target_prompts_tensor != TrainerTools().tokenizer.pad).long()

                max_new_tokens = max(generate_config.max_seq_len - target_prompts_tensor.shape[1], 1)
                completions = []
                actual_chunk_size = len(target_prompts_tensor) if self.chunk_size <= 0 else self.chunk_size

                for chunk_start in range(0, len(target_prompts_tensor), actual_chunk_size):
                    chunk_prompts = target_prompts_tensor[chunk_start: chunk_start + actual_chunk_size]
                    chunk_mask = attention_mask[chunk_start: chunk_start + actual_chunk_size]

                    out_ids, _ = batch_generate(
                        model=self.gen_model,
                        tokens=chunk_prompts,
                        attention_mask=chunk_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=generate_config.temperature,
                        top_k=generate_config.top_k,
                        top_p=generate_config.top_p,
                        repetition_penalty=generate_config.repetition_penalty,
                        exclude_penalty_tokens=generate_config.exclude_penalty_tokens,
                        suppress_tokens=generate_config.suppress_tokens,
                        device=self.generation_device,
                        return_logits=False,
                        auto_prefix_cache=generate_config.auto_prefix_cache
                    )

                    prompt_len = chunk_prompts.shape[1]
                    completions.extend(out_ids[:, prompt_len:].cpu().tolist())

                gathered_results[target_rank] = completions

        if self.world_size > 1:
            results_container = [gathered_results]
            dist.broadcast_object_list(results_container, src=0)
            gathered_results = results_container[0]

        return gathered_results[self.rank]


class ParallelGenerationService(GenerationServiceBase):
    def __init__(
            self,
            model_config: ModelConfig,
            generation_device_mapping: Mapping[str, str],
            chunk_size: int = -1
    ):
        self.generation_device = generation_device_mapping[TrainerTools().parallel.device]
        self.chunk_size = chunk_size
        self.world_size = TrainerTools().parallel.world_size
        self.rank = TrainerTools().parallel.global_rank

        self.gen_model = LlmModel(model_config).to(self.generation_device, get_dtype())
        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def _sync_weights(self, state_dict: Optional[dict], bucket_size_mb: int = 128):
        """
        使用 dist.broadcast，避免broadcast_object_list的序列化操作
        """
        if self.world_size == 1:
            if state_dict is not None:
                self.gen_model.load_state_dict(state_dict, strict=False)
            return

        comm_device = TrainerTools().parallel.device
        gen_device = self.generation_device

        buckets = {}
        bucket_offsets = {}
        params_in_buckets = {}

        def get_bucket(dtype: torch.dtype) -> Tuple[torch.Tensor, int, list]:
            if dtype not in buckets:
                element_size = torch.tensor([], dtype=dtype).element_size()
                bucket_numel = (bucket_size_mb * 1024 * 1024) // element_size
                buckets[dtype] = torch.empty(bucket_numel, dtype=dtype, device=comm_device)
                bucket_offsets[dtype] = 0
                params_in_buckets[dtype] = []
            return buckets[dtype], bucket_offsets[dtype], params_in_buckets[dtype]

        def flush_bucket(dtype: torch.dtype):
            buffer = buckets[dtype]
            offset = bucket_offsets[dtype]
            params_list = params_in_buckets[dtype]

            if offset == 0:
                return

            valid_buffer = buffer[:offset]
            dist.broadcast(valid_buffer, src=0)

            unpack_offset = 0
            for name, param, numel, param_shape in params_list:
                param_slice = valid_buffer[unpack_offset: unpack_offset + numel].view(param_shape)
                param.data.copy_(param_slice.to(gen_device, non_blocking=True))
                unpack_offset += numel

            bucket_offsets[dtype] = 0
            params_list.clear()

        for name, param in self.gen_model.state_dict().items():
            dtype = param.dtype
            numel = param.numel()
            buffer, offset, params_list = get_bucket(dtype)

            if numel > buffer.numel():
                flush_bucket(dtype)
                if TrainerTools().parallel.is_main_process:
                    src_tensor = state_dict[name] if (state_dict is not None and name in state_dict) else param.data
                    src_tensor = src_tensor.to(comm_device, non_blocking=True)
                else:
                    src_tensor = torch.empty_like(param.data, device=comm_device)
                dist.broadcast(src_tensor, src=0)
                param.data.copy_(src_tensor.to(gen_device, non_blocking=True))
                continue

            if offset + numel > buffer.numel():
                flush_bucket(dtype)
                buffer, offset, params_list = get_bucket(dtype)

            if TrainerTools().parallel.is_main_process:
                src_tensor = state_dict[name] if (state_dict is not None and name in state_dict) else param.data
                buffer[offset: offset + numel].copy_(src_tensor.view(-1))

            params_list.append((name, param, numel, param.shape))
            bucket_offsets[dtype] += numel

        for dtype in buckets.keys():
            flush_bucket(dtype)

    def _eval(
            self,
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            generate_config: GenerateConfig,
            task_type: str,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ):
        state_dict = self._get_clean_state_dict_on_rank0(model)
        if TrainerTools().parallel.is_main_process:
            self.gen_model.load_state_dict(state_dict)

            del state_dict
            gc.collect()
            empty_cache()

            max_new_tokens = max(generate_config.max_seq_len - prompt_ids.shape[1], 1)
            out_ids = generate(
                self.gen_model,
                prompt=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=generate_config.temperature,
                top_k=generate_config.top_k,
                top_p=generate_config.top_p,
                repetition_penalty=generate_config.repetition_penalty,
                exclude_penalty_tokens=generate_config.exclude_penalty_tokens,
                suppress_tokens=generate_config.suppress_tokens,
                pixel_values=pixel_values,
                tokens_per_image=tokens_per_image,
                device=self.generation_device,
                return_token=True
            )

            prompt_len = prompt_ids.shape[1]
            return [out_ids[prompt_len:].cpu().tolist()]

        return None

    def _generate(
            self,
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            generate_config: GenerateConfig,
            task_type: str,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ):
        state_dict = self._get_clean_state_dict_on_rank0(model)
        self._sync_weights(state_dict)

        del state_dict
        gc.collect()
        empty_cache()

        local_prompts_tensor = prompt_ids.to(self.generation_device)
        if local_prompts_tensor.numel() == 0:
            return []

        attention_mask = (local_prompts_tensor != TrainerTools().tokenizer.pad).long()
        max_new_tokens = max(generate_config.max_seq_len - local_prompts_tensor.shape[1], 1)
        completions = []
        actual_chunk_size = len(local_prompts_tensor) if self.chunk_size <= 0 else self.chunk_size

        for chunk_start in range(0, len(local_prompts_tensor), actual_chunk_size):
            chunk_prompts = local_prompts_tensor[chunk_start: chunk_start + actual_chunk_size]
            chunk_mask = attention_mask[chunk_start: chunk_start + actual_chunk_size]

            out_ids, _ = batch_generate(
                model=self.gen_model,
                tokens=chunk_prompts,
                attention_mask=chunk_mask,
                max_new_tokens=max_new_tokens,
                temperature=generate_config.temperature,
                top_k=generate_config.top_k,
                top_p=generate_config.top_p,
                repetition_penalty=generate_config.repetition_penalty,
                exclude_penalty_tokens=generate_config.exclude_penalty_tokens,
                suppress_tokens=generate_config.suppress_tokens,
                device=self.generation_device,
                return_logits=False,
                auto_prefix_cache=generate_config.auto_prefix_cache
            )

            prompt_len = chunk_prompts.shape[1]
            completions.extend(out_ids[:, prompt_len:].cpu().tolist())

        return completions

    def __call__(
            self,
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            generate_config: GenerateConfig,
            task_type: str,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ) -> List[List[int]]:
        if task_type == 'eval':
            return self._eval(model, prompt_ids, generate_config, task_type, pixel_values, tokens_per_image)

        return self._generate(model, prompt_ids, generate_config, task_type, pixel_values, tokens_per_image)


class EnvironmentStep(Protocol):
    """
    多轮 RL 交互环境协议。
    """
    def __call__(self, generated_text: str) -> Tuple[bool, str]:
        """
        根据模型生成的文本执行环境交互（如：运行代码、验证公式、调用 API）。

        Args:
            generated_text (str): 模型在当前轮次生成的文本内容。

        Returns:
            Tuple[bool, str]:
                - is_done (bool): 交互是否结束（例如：代码执行成功，或者彻底失败不给重试机会）。
                - feedback (str): 环境返回的反馈信息（例如：编译报错内容）。
        """
        ...

class FeedbackFormatter(Protocol):
    """
    环境反馈格式化协议。
    用于将环境反馈包装成对话格式（如加入特定的 <user> 标识符）。
    """
    def __call__(self, feedback: str) -> str: ...


class MultiTurnRLGenerationService(GenerationServiceBase):
    def __init__(
            self,
            env_step: EnvironmentStep,
            format_feedback: FeedbackFormatter,
            max_turns: int = 3,
            max_consecutive_errors: int = 3
    ):
        self.env_step = env_step
        self.format_feedback = format_feedback
        self.max_turns = max_turns
        self.max_consecutive_errors = max_consecutive_errors

    def __call__(
            self,
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            generate_config: GenerateConfig,
            task_type: str,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ) -> Dict[str, Any]:

        tokenizer = TrainerTools().tokenizer
        device = TrainerTools().parallel.device
        pad_token_id = tokenizer.pad

        batch_size = prompt_ids.shape[0]
        current_prompts = prompt_ids
        dones = [False] * batch_size
        consecutive_errors = [0] * batch_size

        trajectories = [[] for _ in range(batch_size)]
        generation_masks = [[] for _ in range(batch_size)]

        with torch.no_grad(), unwrap_model_for_generation(model) as unwrapped_model:
            gen_model = getattr(unwrapped_model, 'policy_model', unwrapped_model)

            for turn in range(self.max_turns):
                local_all_done = torch.tensor(1 if all(dones) else 0, device=device)
                if TrainerTools().parallel.world_size > 1:
                    dist.all_reduce(local_all_done, op=dist.ReduceOp.SUM)

                if local_all_done.item() == TrainerTools().parallel.world_size:
                    break

                attention_mask = (current_prompts != pad_token_id).long()
                cur_prompt_len = current_prompts.shape[1]
                remaining_max_tokens = max(generate_config.max_seq_len - cur_prompt_len, 1)

                outputs, _ = batch_generate(
                    model=gen_model,
                    tokens=current_prompts,
                    attention_mask=attention_mask,
                    max_new_tokens=remaining_max_tokens,
                    temperature=generate_config.temperature,
                    top_p=generate_config.top_p,
                    top_k=generate_config.top_k,
                    repetition_penalty=generate_config.repetition_penalty,
                    exclude_penalty_tokens=generate_config.exclude_penalty_tokens,
                    suppress_tokens=generate_config.suppress_tokens,
                    device=device,
                    pixel_values=pixel_values,
                    tokens_per_image=tokens_per_image,
                    auto_prefix_cache=generate_config.auto_prefix_cache,
                    return_logits=False
                )

                next_prompts_list = [None] * batch_size
                tasks = []

                for i in range(batch_size):
                    if dones[i]:
                        next_prompts_list[i] = current_prompts[i]
                        continue

                    cur_prompt_len = current_prompts.shape[1]
                    new_tokens = outputs[i, cur_prompt_len:]

                    valid_new_tokens = new_tokens[new_tokens != pad_token_id].tolist()
                    generated_text = tokenizer.decode(valid_new_tokens)

                    trajectories[i].extend(valid_new_tokens)
                    generation_masks[i].extend([True] * len(valid_new_tokens))  # 标记为模型生成

                    tasks.append((i, valid_new_tokens, generated_text))

                results_dict = {}
                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(tasks))) as executor:
                    future_to_idx = {
                        executor.submit(self.env_step, text): (idx, toks)
                        for (idx, toks, text) in tasks
                    }
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx, valid_new_tokens = future_to_idx[future]
                        try:
                            is_done, feedback = future.result()
                            is_exception = False
                        except Exception as e:
                            is_done, feedback = False, f"Error: {str(e)}"
                            is_exception = True

                        results_dict[idx] = (is_done, feedback, valid_new_tokens, is_exception)

                for idx, (is_done, feedback, valid_new_tokens, is_exception) in results_dict.items():
                    if is_exception:
                        consecutive_errors[idx] += 1
                    else:
                        consecutive_errors[idx] = 0

                    if consecutive_errors[idx] >= self.max_consecutive_errors:
                        is_done = True

                    if is_done or turn == self.max_turns - 1:
                        dones[idx] = True
                        next_prompts_list[idx] = current_prompts[idx]
                    else:
                        feedback_text = self.format_feedback(feedback)
                        feedback_tokens = tokenizer.encode(feedback_text)

                        trajectories[idx].extend(feedback_tokens)
                        generation_masks[idx].extend([False] * len(feedback_tokens))

                        unpadded_prompt = current_prompts[idx][current_prompts[idx] != pad_token_id]
                        next_prompt = torch.cat([
                            unpadded_prompt,
                            torch.tensor(valid_new_tokens + feedback_tokens, dtype=torch.long, device=device)
                        ])
                        next_prompts_list[idx] = next_prompt

                if not all(dones):
                    reversed_prompts = [p.flip(dims=(0,)) for p in next_prompts_list]
                    padded_reversed = torch.nn.utils.rnn.pad_sequence(
                        reversed_prompts, batch_first=True, padding_value=pad_token_id
                    )
                    current_prompts = padded_reversed.flip(dims=(1,))

        return {
            'completions': trajectories,
            'dones': dones,
            'generation_masks': generation_masks
        }

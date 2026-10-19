"""
大模型分布式训练生成服务模块 (Generation Services)

【设计背景与核心价值】：
在强化学习 (PPO/GRPO) 采样 Rollout 与评估阶段，模型需要自回归逐 Token 采样生成。
如果模型被 DeepSpeed ZeRO-3 等分布式框架封装，参数被切分在多张 GPU 上，在训练图上直接逐 Token 生成
会导致每一层频繁触发高开销的跨卡 All-Gather。本模块通过定义统一的 GenerationService 协议，
提供多种解耦的生成服务实现：
1. `SyncCentralGenerationService`: 中心化单卡集中同步生成，单点维护非切分副本，最大化节省显存。
2. `ParallelGenerationService`: 分布式多卡并行生成，128MB 桶式底层广播极速同步，高吞吐并行采样。
3. `MultiTurnRLGenerationService`: 多轮强化学习/Agent 环境交互生成服务，统一支持原生 In-Place 原地解包
   与 Parallel 独立推理副本双模式（传入 `model_config` 即可激活多卡独立副本与跨卡卸载，彻底杜绝 ZeRO-3 显存复合爆炸），
   提供多线程环境并发、超时熔断、动态 Padding 与精准 generation_masks 梯度隔离。
"""

from typing import Optional, Mapping, Tuple, Protocol, Dict, Any, List
import gc
import concurrent.futures
import inspect
from contextlib import contextmanager

import torch
import torch.distributed as dist
from llm_model import LlmModel, ModelConfig, VLMConfig, VlmModel

from .tools import TrainerTools
from .train_configs import GenerateConfig
from .utils import get_dtype, empty_cache, batch_repeat_image_tok
from .generate_utils import batch_generate, generate
from .partition_utils import get_full_state_dict_on_rank0, unwrap_model,  unwrap_model_for_generation


class GenerationServiceBase:
    """
    生成服务基类。
    提供大模型分布式环境下权重提取与高效多卡广播同步的基础通用能力。
    """
    def _get_clean_state_dict_on_rank0(self, model: torch.nn.Module) -> Optional[Dict[str, torch.Tensor]]:
        """
        从分布式训练模型中安全提取 Rank 0 上的完整、未切分权重（State Dict），
        并自动剥离可能存在的训练封装前缀（如 'policy_model.'）。

        Args:
            model (torch.nn.Module): 当前训练中的模型实例（可能被 DeepSpeed ZeRO-3 或 FSDP 封装）。

        Returns:
            Optional[Dict[str, torch.Tensor]]:
                - Rank 0: 返回清理后的完整纯净模型 state_dict；
                - 非主卡 (Rank > 0): 返回 None。
        """
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

    @torch.no_grad()
    def _sync_weights(self, state_dict: Optional[dict], bucket_size_mb: int = 128):
        """
        高性能桶式张量广播同步机制。
        使用底层集合通信 `dist.broadcast` 代替慢速的 Python 对象序列化广播，
        将 Rank 0 的最新模型参数分批填充入连续内存 Buffer 桶中同步至全网各卡生成模型，
        并支持跨设备（Cross-Device）异构计算流同步保护。

        Args:
            state_dict (Optional[dict]): Rank 0 传入的待同步完整模型参数字典，其他卡传 None。
            bucket_size_mb (int, optional): 单个通信缓冲桶的容量上限（MB），默认为 128MB。
        """
        world_size = getattr(self, 'world_size', TrainerTools().parallel.world_size)
        if world_size == 1:
            if state_dict is not None and getattr(self, 'gen_model', None) is not None:
                self.gen_model.load_state_dict(state_dict, strict=False)
            return

        comm_device = TrainerTools().parallel.device
        gen_device = getattr(self, 'generation_device', comm_device)
        is_cross_device = torch.device(comm_device) != torch.device(gen_device)

        buckets = {}
        bucket_offsets = {}
        params_in_buckets = {}

        def _synchronize_cross_device():
            if not is_cross_device:
                return
            for dev in (comm_device, gen_device):
                dev_type = torch.device(dev).type
                if dev_type == 'cuda' and torch.cuda.is_available():
                    torch.cuda.synchronize(dev)
                elif dev_type == 'npu' and hasattr(torch, 'npu') and torch.npu.is_available():
                    torch.npu.synchronize(dev)
                elif dev_type == 'mlu' and hasattr(torch, 'mlu') and torch.mlu.is_available():
                    torch.mlu.synchronize(dev)
                elif dev_type == 'mps' and hasattr(torch, 'mps') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.synchronize()

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
                param.data.copy_(param_slice.to(gen_device))
                unpack_offset += numel

            _synchronize_cross_device()

            bucket_offsets[dtype] = 0
            params_list.clear()

        for name, param in sorted(self.gen_model.state_dict().items(), key=lambda x: x[0]):
            dtype = param.dtype
            numel = param.numel()
            buffer, offset, params_list = get_bucket(dtype)

            if numel > buffer.numel():
                flush_bucket(dtype)
                if TrainerTools().parallel.is_main_process:
                    src_tensor = state_dict[name] if (state_dict is not None and name in state_dict) else param.data
                    src_tensor = src_tensor.to(comm_device, dtype=param.dtype).contiguous()
                else:
                    src_tensor = torch.empty_like(param.data, device=comm_device)
                dist.broadcast(src_tensor, src=0)
                param.data.copy_(src_tensor.to(gen_device))
                _synchronize_cross_device()
                continue

            if offset + numel > buffer.numel():
                flush_bucket(dtype)
                buffer, offset, params_list = get_bucket(dtype)

            if TrainerTools().parallel.is_main_process:
                src_tensor = state_dict[name] if (state_dict is not None and name in state_dict) else param.data
                buffer[offset: offset + numel].copy_(src_tensor.reshape(-1))

            params_list.append((name, param, numel, param.shape))
            bucket_offsets[dtype] += numel

        for dtype in sorted(buckets.keys(), key=str):
            flush_bucket(dtype)


class SyncCentralGenerationService(GenerationServiceBase):
    """
    中心化集中式同步生成服务 (Centralized Synchronous Generation Service)。

    【设计定位与工作原理】：
    1. 专为单机多卡显存极度紧张场景设计。仅在 Rank 0 维护一份非切分的独立推理模型副本（self.gen_model），
       其他非主卡（Rank > 0）完全不占用常驻推理显存（self.gen_model 为 None）。
    2. 生成调用时，各卡通过 `dist.all_gather_object` 将本地 Prompts 统一汇总汇聚至主卡 Rank 0。
    3. Rank 0 在其独立推理设备上对汇总的所有 Prompts 进行分块（chunked_generate_size）批量推理。
    4. 推理完成后，Rank 0 将汇总结果通过 `dist.broadcast_object_list` 广播分发回各卡，各卡取回属于自己本地 Batch 的生成结果。

    【适用场景】：
    - 各卡显存紧张、无法额外容纳一份完整独立推理模型时；
    - 仅单卡具备额外大显存或专用推理卡的场景。
    """
    def __init__(
            self,
            model_config: ModelConfig,
            generation_device: str
    ):
        """
        初始化中心化集中式生成服务。

        Args:
            model_config (ModelConfig): 模型结构配置对象（ModelConfig 或 VLMConfig），用于构建 Rank 0 推理模型。
            generation_device (str): Rank 0 上执行生成任务的目标设备（例如 'cuda:0' 或独立推理卡 'cuda:1'）。
        """
        self.generation_device = generation_device
        self.world_size = TrainerTools().parallel.world_size
        self.rank = TrainerTools().parallel.global_rank

        if TrainerTools().parallel.is_main_process:
            if isinstance(model_config, VLMConfig):
                self.gen_model = VlmModel(model_config).to(self.generation_device, get_dtype())
            else:
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
    ) -> Dict[str, Any]:
        """
        执行集中式单卡推理生成。

        Args:
            model (torch.nn.Module): 当前训练中的模型实例（从中提取最新权重并加载到 Rank 0 的 gen_model 中）。
            prompt_ids (torch.Tensor): 本地 Rank 待生成的输入 Prompt Token IDs，形状为 [batch_size, seq_len]。
            generate_config (GenerateConfig): 生成超参数配置（max_seq_len, temperature, top_p, top_k 等）。
            task_type (str): 调用任务上下文（如 'eval', 'ppo', 'grpo'）。若为 'eval' 则仅处理 Rank 0 避免重复。
            pixel_values (Optional[torch.Tensor], optional): 多模态图像特征张量，仅 VLM 场景有效。
            tokens_per_image (Optional[int], optional): 单张图像对应的虚拟 Token 数量，仅 VLM 场景有效。

        Returns:
            Dict[str, Any]:
                - 'completions' (List[List[int]]): 当前 Rank 对应样本的新生成 Token ID 序列列表。
        """
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

                if isinstance(self.gen_model, VlmModel) and tokens_per_image is not None and tokens_per_image > 1:
                    target_prompts_tensor, attention_mask = batch_repeat_image_tok(
                        target_prompts_tensor, tokens_per_image, attention_mask
                    )

                max_new_tokens = max(generate_config.max_seq_len - target_prompts_tensor.shape[1], 1)
                completions = []

                actual_chunk_size = generate_config.chunked_generate_size
                if actual_chunk_size is None or actual_chunk_size <= 0:
                    actual_chunk_size = len(target_prompts_tensor)

                for chunk_start in range(0, len(target_prompts_tensor), actual_chunk_size):
                    chunk_prompts = target_prompts_tensor[chunk_start: chunk_start + actual_chunk_size]
                    chunk_mask = attention_mask[chunk_start: chunk_start + actual_chunk_size]

                    chunk_pixel_values = None
                    if pixel_values is not None:
                        chunk_pixel_values = pixel_values[chunk_start: chunk_start + actual_chunk_size]

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
                        pixel_values=chunk_pixel_values,
                        tokens_per_image=-1,
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

        return {
            'completions': gathered_results[self.rank]
        }


class ParallelGenerationService(GenerationServiceBase):
    """
    分布式多卡并行生成服务 (Distributed Parallel Generation Service)。

    【设计定位与工作原理】：
    1. 专为追求高生成吞吐的标准强化学习 (PPO/GRPO) 采样与评估场景设计。
    2. 每张 GPU 维护独立的、未切分的推理模型副本（self.gen_model），并通过 `_sync_weights`
       使用 128MB 桶聚合底层张量广播 (`dist.broadcast`) 高速同步最新训练参数，无 Python 序列化开销。
    3. 支持通过 `generation_device_mapping` 将生成卸载到独立卡（例如训练在 gpu:0，生成在 gpu:2），
       实现训练卡与推理卡的物理分离，大幅规避显存竞争。
    4. 权重同步完成后，各卡本地独立执行分块批量推理（batch_generate），互不阻塞，生成吞吐呈线性扩展。

    【适用场景】：
    - 显存充裕的多卡分布式训练环境；
    - 单轮 PPO/GRPO 采样与评测的高吞吐场景。
    """
    def __init__(
            self,
            model_config: ModelConfig,
            generation_device_mapping: Optional[Mapping[str, str]] = None
    ):
        """
        初始化分布式多卡并行生成服务。

        Args:
            model_config (ModelConfig): 模型结构配置对象（ModelConfig 或 VLMConfig），用于构建各卡本地的纯净推理副本。
            generation_device_mapping (Optional[Mapping[str, str]], optional):
                训练卡设备到生成卡设备的映射字典。例如 `{"cuda:0": "cuda:2", "cuda:1": "cuda:3"}`。
                若为 None，则默认各卡在本地相同的训练设备上执行生成。
        """
        if generation_device_mapping is not None:
            self.generation_device = generation_device_mapping[TrainerTools().parallel.device]
        else:
            self.generation_device = TrainerTools().parallel.device
        self.world_size = TrainerTools().parallel.world_size
        self.rank = TrainerTools().parallel.global_rank

        if isinstance(model_config, VLMConfig):
            self.gen_model = VlmModel(model_config).to(self.generation_device, get_dtype())
        else:
            self.gen_model = LlmModel(model_config).to(self.generation_device, get_dtype())

        self.gen_model.eval()
        for param in self.gen_model.parameters():
            param.requires_grad = False

    def _eval(
            self,
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            generate_config: GenerateConfig,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ) -> List[List[int]]:
        """
        评测阶段的单样本/批次生成（内部方法）。
        先通过桶式广播同步权重，然后在本地使用 generate 进行采样。

        Args:
            model (torch.nn.Module): 当前训练模型。
            prompt_ids (torch.Tensor): 输入 Prompt Token IDs。
            generate_config (GenerateConfig): 生成超参数配置。
            pixel_values (Optional[torch.Tensor], optional): 多模态特征张量。
            tokens_per_image (Optional[int], optional): 单张图像虚拟 Token 数。

        Returns:
            List[List[int]]: 生成完成的 Token ID 列表。
        """
        state_dict = self._get_clean_state_dict_on_rank0(model)
        self._sync_weights(state_dict)

        del state_dict
        gc.collect()
        empty_cache()

        local_prompts_tensor = prompt_ids.to(self.generation_device)
        if tokens_per_image is not None and tokens_per_image > 1:
            from .utils import batch_repeat_image_tok
            local_prompts_tensor, _ = batch_repeat_image_tok(local_prompts_tensor, tokens_per_image, None)
            tokens_per_image = -1

        max_new_tokens = max(generate_config.max_seq_len - local_prompts_tensor.shape[1], 1)
        with torch.no_grad():
            out_ids = generate(
                self.gen_model,
                prompt=local_prompts_tensor,
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

        prompt_len = local_prompts_tensor.shape[1]
        return [out_ids[prompt_len:].cpu().tolist()]

    def _generate(
            self,
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            generate_config: GenerateConfig,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ) -> List[List[int]]:
        """
        强化学习采样阶段的分块批量并行生成（内部方法）。
        先通过桶式广播同步权重，然后按 chunked_generate_size 执行 batch_generate。

        Args:
            model (torch.nn.Module): 当前训练模型。
            prompt_ids (torch.Tensor): 输入 Prompt Token IDs。
            generate_config (GenerateConfig): 生成超参数配置。
            pixel_values (Optional[torch.Tensor], optional): 多模态特征张量。
            tokens_per_image (Optional[int], optional): 单张图像虚拟 Token 数。

        Returns:
            List[List[int]]: 本地批次各样本生成的 Completion Token ID 序列列表。
        """
        state_dict = self._get_clean_state_dict_on_rank0(model)
        self._sync_weights(state_dict)

        del state_dict
        gc.collect()
        empty_cache()

        local_prompts_tensor = prompt_ids.to(self.generation_device)
        if tokens_per_image is not None and tokens_per_image > 1:
            from .utils import batch_repeat_image_tok
            local_prompts_tensor, _ = batch_repeat_image_tok(local_prompts_tensor, tokens_per_image, None)
            tokens_per_image = -1

        if local_prompts_tensor.numel() == 0:
            return []

        attention_mask = (local_prompts_tensor != TrainerTools().tokenizer.pad).long()
        max_new_tokens = max(generate_config.max_seq_len - local_prompts_tensor.shape[1], 1)
        completions = []

        actual_chunk_size = generate_config.chunked_generate_size
        if actual_chunk_size is None or actual_chunk_size <= 0:
            actual_chunk_size = len(local_prompts_tensor)

        for chunk_start in range(0, len(local_prompts_tensor), actual_chunk_size):
            chunk_prompts = local_prompts_tensor[chunk_start: chunk_start + actual_chunk_size]
            chunk_mask = attention_mask[chunk_start: chunk_start + actual_chunk_size]

            if pixel_values is not None:
                chunk_pixel_values = pixel_values[chunk_start: chunk_start + actual_chunk_size]
            else:
                chunk_pixel_values = None

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
                pixel_values=chunk_pixel_values,
                tokens_per_image=tokens_per_image,
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
    ) -> Dict[str, Any]:
        """
        执行分布式多卡并行生成。

        Args:
            model (torch.nn.Module): 当前训练模型。
            prompt_ids (torch.Tensor): 本地批次的 Prompt Token IDs。
            generate_config (GenerateConfig): 生成超参数配置。
            task_type (str): 调用任务上下文（'eval', 'ppo', 'grpo' 等）。
            pixel_values (Optional[torch.Tensor], optional): 多模态图像特征。
            tokens_per_image (Optional[int], optional): 单张图像虚拟 Token 数量。

        Returns:
            Dict[str, Any]:
                - 'completions' (List[List[int]]): 本地各样本生成的 Token ID 序列列表。
        """
        if task_type == 'eval':
            return {
                'completions': self._eval(model, prompt_ids, generate_config, pixel_values, tokens_per_image)
            }

        return {
            'completions': self._generate(model, prompt_ids, generate_config, pixel_values, tokens_per_image)
        }


class EnvironmentStep(Protocol):
    """
    多轮 RL 交互环境协议。
    """
    def __call__(
            self,
            sample_idx: int,
            prompt_ids: List[int],
            history_tokens: List[int],
            generated_text: str
    ) -> Tuple[bool, str]:
        """
        根据模型生成的文本执行环境交互（如：运行代码、验证公式、调用 API）。

        Args:
            sample_idx (int): 样本在当前批次中的索引。
            prompt_ids (List[int]): 样本初始 Prompt Token IDs（去除 padding）。
            history_tokens (List[int]): 之前轮次累积的历史 Token IDs（包含历史生成内容与环境反馈）。
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
    """
    多轮强化学习 / Agent 环境交互生成服务 (Multi-Turn RL Generation Service)。

    【设计定位与工作原理】：
    专为代码执行 (Code RL)、数学证明验证、工具调用与多轮对话等智能体交互任务设计。
    提供完整的交互状态机驱动、动态 Padding、分块推理与精准梯度掩码支持：

    1. 双模式后端解耦 (In-Place vs Parallel)：
       - 原生模式 (model_config is None): 通过 `unwrap_model_for_generation` 原地解包训练模型，
         无需额外常驻副本显存，适合小规模轻量任务。
       - 并行模式 (model_config is not None): 每张卡持有独立未切分的纯净推理副本 (self.gen_model)，
         使用 128MB 桶聚合广播 (`_sync_weights`) 极速同步最新参数；支持通过 `generation_device_mapping`
         将生成任务卸载到独立卡，彻底根除 ZeRO-3 下 GatheredParameters 叠加多轮长序列 KV Cache 的复合 OOM 隐患。
    2. 多轮 Rollout 循环状态机 (max_turns)：
       - 跨卡同步协同：通过 `dist.all_reduce` 监测全网终止状态，所有卡样本全结束时同步退出，杜绝死锁。
       - 活跃样本过滤：每轮仅聚合未结束样本（active indices）组 Batch 推理，降低无谓显存与算力浪费。
    3. 异步多线程环境步进与超时保护：
       - 使用 ThreadPoolExecutor 并发执行环境回调 `env_step`（如运行代码解释器、API 请求）。
       - 内置 `env_timeout` 超时强杀机制（防止外部环境/沙箱死循环导致训练卡死）与 `max_consecutive_errors` 熔断机制。
    4. 严格预算控制与悬空反馈丢弃：
       - 动态校验整条轨迹预算容量，若追加环境反馈后无法继续生成下一步动作，则直接终止并不拼接悬空反馈，
         确保与 Trainer 端 PPO/GRPO 的 GAE Bootstrap 对齐严格一致。
    5. 精准 generation_masks 构建（核心机制）：
       - 自动为轨迹中模型生成的动作 Token 打上 True，为系统环境反馈拼接的 Token 打上 False。
       - Trainer 据此构建 `loss_mask = completion_pad_mask & external_gen_mask`，反向传播时仅对模型
         自身生成的 Token 计算策略梯度与 KL 散度，严密屏蔽环境反馈内容。
    """
    def __init__(
            self,
            env_step: EnvironmentStep,
            format_feedback: FeedbackFormatter,
            model_config: Optional[ModelConfig] = None,
            generation_device_mapping: Optional[Mapping[str, str]] = None,
            max_turns: int = 3,
            max_consecutive_errors: int = 3,
            env_timeout: Optional[float] = 30.0,
            max_workers: Optional[int] = None
    ):
        """
        初始化多轮强化学习生成服务。

        Args:
            env_step (EnvironmentStep):
                环境步进交互回调函数。签名支持：
                - `(sample_idx, prompt_ids, history_tokens, generated_text) -> (is_done, feedback)`
                - 或简化版单个入参 `(generated_text) -> (is_done, feedback)`
            format_feedback (FeedbackFormatter):
                环境反馈文本格式化回调函数，负责将环境反馈包装进 Chat Template 对话格式（如 `<user>...</s><assistant>`）。
            model_config (Optional[ModelConfig], optional):
                模型架构配置。若传入，则自动启用 Parallel 独立副本并行模式（创建独立推理模型并使用桶式张量同步）；
                若为 None，则保持 In-Place 原地解包模式。默认为 None。
            generation_device_mapping (Optional[Mapping[str, str]], optional):
                训练设备到生成设备的映射字典（如 `{"cuda:0": "cuda:2"}`）。仅在传入 model_config 时生效。
                若为 None 且开启了并行模式，则默认使用各卡本地对应的训练设备。默认为 None。
            max_turns (int, optional):
                每个样本允许交互的最大轮次上限（Turn Limit），默认为 3。
            max_consecutive_errors (int, optional):
                单个样本允许连续发生环境异常/超时的最大次数，超过该阈值时自动熔断并提前终止该样本，默认为 3。
            env_timeout (Optional[float], optional):
                单个环境步进操作的最大超时秒数。超时后自动判定为异常并返回 TimeoutError，防止训练被沙箱挂死。默认为 30.0 秒。
            max_workers (Optional[int], optional):
                执行 env_step 异步多线程池的最大并发 Worker 数量。默认为 None（自动自适应样本数量与核数）。
        """
        self.env_step = env_step
        self.format_feedback = format_feedback
        self.max_turns = max_turns
        self.max_consecutive_errors = max_consecutive_errors
        self.env_timeout = env_timeout
        self.max_workers = max_workers

        self.model_config = model_config
        self.generation_device_mapping = generation_device_mapping
        self.is_parallel = (model_config is not None)

        self.world_size = TrainerTools().parallel.world_size
        self.rank = TrainerTools().parallel.global_rank

        if self.is_parallel:
            if generation_device_mapping is not None:
                self.generation_device = generation_device_mapping[TrainerTools().parallel.device]
            else:
                self.generation_device = TrainerTools().parallel.device

            if isinstance(model_config, VLMConfig):
                self.gen_model = VlmModel(model_config).to(self.generation_device, get_dtype())
            else:
                self.gen_model = LlmModel(model_config).to(self.generation_device, get_dtype())

            self.gen_model.eval()
            for param in self.gen_model.parameters():
                param.requires_grad = False
        else:
            self.generation_device = TrainerTools().parallel.device
            self.gen_model = None

        self._env_step_accepts_context = True
        try:
            sig = inspect.signature(self.env_step)
            pos_params = [
                p for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            if len(pos_params) == 1:
                self._env_step_accepts_context = False
        except Exception: ...

    @contextmanager
    def _get_generation_model_context(self, model: torch.nn.Module):
        """
        生成模型与设备获取上下文管理器（内部方法）。
        根据当前服务处于 Parallel 模式还是 In-Place 模式，动态提供适宜的前向模型与执行设备：
        - Parallel 模式：提取 Rank 0 权重并通过 128MB 桶广播同步至 self.gen_model，挂载到 self.generation_device；
        - In-Place 模式：调用 unwrap_model_for_generation(model) 解包训练模型并管理 eval/train 状态流转。

        Args:
            model (torch.nn.Module): 当前训练阶段的模型实例。

        Yields:
            Tuple[torch.nn.Module, str]: (用于推理的模型实例, 推理执行所在的目标设备)。
        """
        if self.is_parallel:
            state_dict = self._get_clean_state_dict_on_rank0(model)
            self._sync_weights(state_dict)
            del state_dict
            gc.collect()
            empty_cache()
            with torch.no_grad():
                yield self.gen_model, self.generation_device
        else:
            comm_device = TrainerTools().parallel.device
            with torch.no_grad(), unwrap_model_for_generation(model) as unwrapped_model:
                gen_model = getattr(unwrapped_model, 'policy_model', unwrapped_model)
                was_training = unwrapped_model.training
                unwrapped_model.eval()
                try:
                    yield gen_model, comm_device
                finally:
                    if was_training:
                        unwrapped_model.train()

    def _call_env_step(
            self,
            sample_idx: int,
            prompt_ids: List[int],
            history_tokens: List[int],
            generated_text: str
    ) -> Tuple[bool, str]:
        """
        根据用户注入的 env_step 签名动态适配入参，安全执行单步环境交互。

        Args:
            sample_idx (int): 样本在当前 Batch 中的索引。
            prompt_ids (List[int]): 样本初始去除 Padding 后的 Prompt Token IDs。
            history_tokens (List[int]): 之前轮次累积的完整历史 Token IDs（包含模型输出与环境反馈）。
            generated_text (str): 模型在当前轮次生成的解码文本。

        Returns:
            Tuple[bool, str]: (is_done 终止标志, feedback 环境反馈文本)。
        """
        if self._env_step_accepts_context:
            return self.env_step(sample_idx, prompt_ids, history_tokens, generated_text)
        return self.env_step(generated_text)  # type: ignore

    def _safe_left_unpad(self, seq: torch.Tensor, pad_id: int) -> torch.Tensor:
        """
        安全剥离 1D 序列左侧的所有连续 Padding Token，返回首个有效 Token 起始的纯净子张量。

        Args:
            seq (torch.Tensor): 待处理的 1D Token 张量。
            pad_id (int): 填充 Token ID。

        Returns:
            torch.Tensor: 去除左侧 Padding 后的 Token 张量。
        """
        non_pad_mask = seq != pad_id
        if non_pad_mask.any():
            first_valid_idx = non_pad_mask.nonzero(as_tuple=True)[0][0]
            return seq[first_valid_idx:]
        return torch.empty(0, dtype=torch.long, device=seq.device)

    def _eval(
            self,
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            generate_config: GenerateConfig,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ) -> Dict[str, Any]:
        """
        专用于 BaseTrainer._eval 阶段的纯文本生成，绝不触发 self.env_step 环境交互。
        所有 Rank 同步参与，保证 ZeRO-3 集合通信严格对齐。

        Args:
            model (torch.nn.Module): 当前训练模型。
            prompt_ids (torch.Tensor): 输入 Prompt Token IDs。
            generate_config (GenerateConfig): 生成超参数配置。
            pixel_values (Optional[torch.Tensor], optional): 多模态图像特征。
            tokens_per_image (Optional[int], optional): 单张图像虚拟 Token 数量。

        Returns:
            Dict[str, Any]: 评测结果字典，包含 completions, dones, generation_masks 与 feedbacks。
        """
        tokenizer = TrainerTools().tokenizer
        pad_token_id = tokenizer.pad

        with self._get_generation_model_context(model) as (gen_model, gen_device):
            prompt_ids = prompt_ids.to(gen_device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(gen_device)

            if tokens_per_image is not None and tokens_per_image > 1:
                from .utils import batch_repeat_image_tok
                prompt_ids, _ = batch_repeat_image_tok(prompt_ids, tokens_per_image, None)
                tokens_per_image = -1

            attention_mask = (prompt_ids != pad_token_id).long()
            if (attention_mask == 0).all():
                attention_mask[:, -1] = 1

            prompt_len = prompt_ids.shape[1]
            max_new_tokens = max(generate_config.max_seq_len - prompt_len, 1)

            out_ids, _ = batch_generate(
                model=gen_model,
                tokens=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=generate_config.temperature,
                top_p=generate_config.top_p,
                top_k=generate_config.top_k,
                repetition_penalty=generate_config.repetition_penalty,
                exclude_penalty_tokens=generate_config.exclude_penalty_tokens,
                suppress_tokens=generate_config.suppress_tokens,
                device=gen_device,
                pixel_values=pixel_values,
                tokens_per_image=tokens_per_image,
                auto_prefix_cache=generate_config.auto_prefix_cache,
                return_logits=False
            )

        empty_cache()
        completions = out_ids[:, prompt_len:].cpu().tolist()

        return {
            'completions': completions,
            'dones': [True] * len(completions),
            'generation_masks': [[True] * len(c) for c in completions],
            'feedbacks': ["" for _ in completions]
        }

    def __call__(
            self,
            model: torch.nn.Module,
            prompt_ids: torch.Tensor,
            generate_config: GenerateConfig,
            task_type: str,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ) -> Dict[str, Any]:
        """
        多轮强化学习生成服务的完整 Rollout 驱动主流程。
        驱动多轮“模型生成 -> 异步环境执行 -> 反馈格式化 -> 动态 Prompt 构造 -> 下一轮生成”的状态机循环。

        Args:
            model (torch.nn.Module): 当前训练中的模型实例。
            prompt_ids (torch.Tensor): 本地批次的 Prompt Token IDs，形状为 [batch_size, seq_len]。
            generate_config (GenerateConfig): 生成解码控制配置（如 temp, top_p, top_k 等）。
            task_type (str): 调用任务上下文（如 'eval', 'ppo', 'grpo'）。
            pixel_values (Optional[torch.Tensor], optional): 多模态图像特征张量。
            tokens_per_image (Optional[int], optional): 单张图像虚拟 Token 数量。

        Returns:
            Dict[str, Any]:
                - 'completions' (List[List[int]]): 包含完整交互轨迹的 Token ID 序列列表（包含模型生成与环境反馈）。
                - 'dones' (List[bool]): 每个样本最终的环境交互终止状态标志列表。
                - 'generation_masks' (List[List[bool]]): 标记轨迹中模型生成的 Token (True) 与环境反馈 (False) 的布尔掩码。
                - 'feedbacks' (List[str]): 每个样本在最后一轮环境交互中返回的原始文本反馈。
        """
        if task_type == 'eval':
            return self._eval(
                model=model,
                prompt_ids=prompt_ids,
                generate_config=generate_config,
                pixel_values=pixel_values,
                tokens_per_image=tokens_per_image
            )

        tokenizer = TrainerTools().tokenizer
        comm_device = TrainerTools().parallel.device
        pad_token_id = tokenizer.pad

        with self._get_generation_model_context(model) as (gen_model, gen_device):
            prompt_ids = prompt_ids.to(gen_device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(gen_device)

            if tokens_per_image is not None and tokens_per_image > 1:
                from .utils import batch_repeat_image_tok
                prompt_ids, _ = batch_repeat_image_tok(prompt_ids, tokens_per_image, None)
                tokens_per_image = -1

            batch_size = prompt_ids.shape[0]
            current_prompts = prompt_ids
            initial_prompts = [
                self._safe_left_unpad(prompt_ids[i], pad_token_id).cpu().tolist()
                for i in range(batch_size)
            ]

            # 统一以当前 Batch 的最大 Prompt 长度为基准，计算整条轨迹在 Trainer 端允许的最大完成预算
            # 确保与 Trainer 端 left_pad 后的 max_new_tokens = max_seq_len - prompt_len 严格契合，杜绝事后静默截断
            initial_max_prompt_len = prompt_ids.shape[1]
            max_completion_budget = max(generate_config.max_seq_len - initial_max_prompt_len, 1)

            loop_dones = [False] * batch_size
            terminals = [False] * batch_size
            consecutive_errors = [0] * batch_size

            trajectories = [[] for _ in range(batch_size)]
            generation_masks = [[] for _ in range(batch_size)]
            last_feedbacks = ["" for _ in range(batch_size)]

            for turn in range(self.max_turns):
                for i in range(batch_size):
                    if not loop_dones[i]:
                        actual_len = (current_prompts[i] != pad_token_id).sum().item()
                        if actual_len >= generate_config.max_seq_len or len(trajectories[i]) >= max_completion_budget:
                            loop_dones[i] = True

                local_all_done = torch.tensor(1 if all(loop_dones) else 0, device=comm_device)
                if TrainerTools().parallel.world_size > 1:
                    dist.all_reduce(local_all_done, op=dist.ReduceOp.SUM)

                if local_all_done.item() == TrainerTools().parallel.world_size:
                    break

                active_indices = [i for i, d in enumerate(loop_dones) if not d]

                if not active_indices:
                    active_prompts = torch.tensor([[pad_token_id]], dtype=torch.long, device=gen_device)
                    attention_mask = torch.ones((1, 1), dtype=torch.long, device=gen_device)
                    local_remaining_max_tokens = 1
                    active_pixel_values = None
                    cur_prompt_len = 1
                else:
                    active_prompts = current_prompts[active_indices]
                    non_pad_mask = (active_prompts != pad_token_id).any(dim=0)

                    first_valid_idx = non_pad_mask.nonzero(as_tuple=True)[0]
                    if len(first_valid_idx) > 0:
                        active_prompts = active_prompts[:, first_valid_idx[0]:]

                    cur_prompt_len = active_prompts.shape[1]
                    active_pixel_values = None
                    if pixel_values is not None:
                        if pixel_values.shape[0] == batch_size:
                            active_pixel_values = pixel_values[active_indices]
                        else:
                            active_pixel_values = pixel_values

                    attention_mask = (active_prompts != pad_token_id).long()

                    if (attention_mask == 0).all():
                        attention_mask[:, -1] = 1

                    # 计算活跃样本的最大剩余预算，避免生成步数超出序列预算限制
                    max_active_budget = max(
                        (max(max_completion_budget - len(trajectories[i]), 1) for i in active_indices),
                        default=1
                    )
                    local_remaining_max_tokens = min(
                        max(generate_config.max_seq_len - cur_prompt_len, 1),
                        max_active_budget
                    )

                remaining_max_tokens = local_remaining_max_tokens

                actual_chunk_size = generate_config.chunked_generate_size or active_prompts.shape[0]

                if 0 < actual_chunk_size < active_prompts.shape[0]:
                    all_outputs = []
                    for chunk_start in range(0, active_prompts.shape[0], actual_chunk_size):
                        chunk_prompts = active_prompts[chunk_start: chunk_start + actual_chunk_size]
                        chunk_mask = attention_mask[chunk_start: chunk_start + actual_chunk_size]
                        chunk_pixel_values = None
                        if active_pixel_values is not None:
                            chunk_pixel_values = active_pixel_values[chunk_start: chunk_start + actual_chunk_size]

                        chunk_out, _ = batch_generate(
                            model=gen_model,
                            tokens=chunk_prompts,
                            attention_mask=chunk_mask,
                            max_new_tokens=remaining_max_tokens,
                            temperature=generate_config.temperature,
                            top_p=generate_config.top_p,
                            top_k=generate_config.top_k,
                            repetition_penalty=generate_config.repetition_penalty,
                            exclude_penalty_tokens=generate_config.exclude_penalty_tokens,
                            suppress_tokens=generate_config.suppress_tokens,
                            device=gen_device,
                            pixel_values=chunk_pixel_values,
                            tokens_per_image=tokens_per_image,
                            auto_prefix_cache=generate_config.auto_prefix_cache,
                            return_logits=False
                        )
                        all_outputs.append(chunk_out)

                    max_out_len = max(o.shape[1] for o in all_outputs)
                    padded_outputs = []
                    for o in all_outputs:
                        if o.shape[1] < max_out_len:
                            o = torch.nn.functional.pad(o, (0, max_out_len - o.shape[1]), value=pad_token_id)
                        padded_outputs.append(o)
                    outputs = torch.cat(padded_outputs, dim=0)
                else:
                    outputs, _ = batch_generate(
                        model=gen_model,
                        tokens=active_prompts,
                        attention_mask=attention_mask,
                        max_new_tokens=remaining_max_tokens,
                        temperature=generate_config.temperature,
                        top_p=generate_config.top_p,
                        top_k=generate_config.top_k,
                        repetition_penalty=generate_config.repetition_penalty,
                        exclude_penalty_tokens=generate_config.exclude_penalty_tokens,
                        suppress_tokens=generate_config.suppress_tokens,
                        device=gen_device,
                        pixel_values=active_pixel_values,
                        tokens_per_image=tokens_per_image,
                        auto_prefix_cache=generate_config.auto_prefix_cache,
                        return_logits=False
                    )

                empty_cache()

                next_prompts_list = [None] * batch_size
                batch_budget_exhausted = [False] * batch_size
                tasks = []

                if active_indices:
                    for i in range(batch_size):
                        if loop_dones[i]:
                            next_prompts_list[i] = self._safe_left_unpad(current_prompts[i], pad_token_id)

                    for local_idx, global_idx in enumerate(active_indices):
                        if loop_dones[global_idx]:
                            continue

                        new_tokens = outputs[local_idx, cur_prompt_len:]
                        valid_new_tokens = new_tokens[new_tokens != pad_token_id].tolist()

                        rem_budget = max(max_completion_budget - len(trajectories[global_idx]), 0)
                        if len(valid_new_tokens) >= rem_budget:
                            valid_new_tokens = valid_new_tokens[:rem_budget]
                            batch_budget_exhausted[global_idx] = True

                        generated_text = tokenizer.decode(valid_new_tokens)
                        history_tokens = list(trajectories[global_idx])

                        trajectories[global_idx].extend(valid_new_tokens)
                        generation_masks[global_idx].extend([True] * len(valid_new_tokens))

                        tasks.append((global_idx, valid_new_tokens, initial_prompts[global_idx], history_tokens, generated_text))

                results_dict = {}
                if tasks:
                    workers = self.max_workers if (self.max_workers and self.max_workers > 0) else min(32, max(1, len(tasks)))
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
                    try:
                        future_to_idx = {
                            executor.submit(self._call_env_step, idx, p_ids, hist_toks, text): (idx, toks)
                            for (idx, toks, p_ids, hist_toks, text) in tasks
                        }

                        if self.env_timeout is not None and self.env_timeout > 0:
                            done, not_done = concurrent.futures.wait(
                                future_to_idx.keys(),
                                timeout=self.env_timeout
                            )

                            for future in done:
                                idx, valid_new_tokens = future_to_idx[future]
                                try:
                                    is_done, feedback = future.result()
                                    is_exception = False
                                except Exception as e:
                                    is_done, feedback = False, f"Error: {str(e)}"
                                    is_exception = True
                                results_dict[idx] = (is_done, feedback, valid_new_tokens, is_exception)

                            for future in not_done:
                                idx, valid_new_tokens = future_to_idx[future]
                                future.cancel()
                                is_done, feedback = False, f"TimeoutError: Environment step exceeded timeout of {self.env_timeout}s"
                                is_exception = True
                                results_dict[idx] = (is_done, feedback, valid_new_tokens, is_exception)
                        else:
                            for future in concurrent.futures.as_completed(future_to_idx):
                                idx, valid_new_tokens = future_to_idx[future]
                                try:
                                    is_done, feedback = future.result()
                                    is_exception = False
                                except Exception as e:
                                    is_done, feedback = False, f"Error: {str(e)}"
                                    is_exception = True
                                results_dict[idx] = (is_done, feedback, valid_new_tokens, is_exception)
                    finally:
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            executor.shutdown(wait=False)

                for idx, (is_done, feedback, valid_new_tokens, is_exception) in results_dict.items():
                    last_feedbacks[idx] = feedback
                    if len(valid_new_tokens) == 0:
                        is_exception = True

                    if is_exception:
                        consecutive_errors[idx] += 1
                    else:
                        consecutive_errors[idx] = 0

                    if is_done:
                        terminals[idx] = True
                        loop_dones[idx] = True
                    elif batch_budget_exhausted[idx] or consecutive_errors[idx] >= self.max_consecutive_errors:
                        loop_dones[idx] = True

                    # 若样本已经结束交互（任务完成/连续异常/达到上限）或已是最后一轮，无需追加尾部 feedback 也无需构造下一轮 prompt
                    if loop_dones[idx] or turn == self.max_turns - 1:
                        loop_dones[idx] = True
                        next_prompts_list[idx] = torch.tensor([pad_token_id], dtype=torch.long, device=gen_device)
                        continue

                    unpadded_prompt = self._safe_left_unpad(current_prompts[idx], pad_token_id)

                    try:
                        feedback_text = self.format_feedback(feedback)
                        feedback_tokens = tokenizer.encode(feedback_text)
                    except Exception as e:
                        feedback_text = f"\nError formatting feedback: {str(e)}\n"
                        feedback_tokens = tokenizer.encode(feedback_text)

                    remaining_capacity = max_completion_budget - len(trajectories[idx])
                    next_prompt_len = unpadded_prompt.size(0) + len(valid_new_tokens) + len(feedback_tokens)

                    # 若追加环境反馈后将耗尽完成预算或达到最大序列长度，
                    # 说明模型在下一轮已无法继续生成任何动作 Token，此时不应将该段悬空 feedback
                    # 塞入 trajectories 尾部（避免污染 completion_ids 长度并导致 PPO GAE Bootstrap 对齐错位），
                    # 直接终止该样本。
                    if remaining_capacity <= 0 or len(feedback_tokens) >= remaining_capacity or next_prompt_len >= generate_config.max_seq_len:
                        loop_dones[idx] = True
                        next_prompts_list[idx] = torch.tensor([pad_token_id], dtype=torch.long, device=gen_device)
                        continue

                    if feedback_tokens:
                        trajectories[idx].extend(feedback_tokens)
                        generation_masks[idx].extend([False] * len(feedback_tokens))

                    next_prompt = torch.cat([
                        unpadded_prompt,
                        torch.tensor(valid_new_tokens + feedback_tokens, dtype=torch.long, device=gen_device)
                    ])
                    next_prompts_list[idx] = next_prompt

                if not all(loop_dones) and active_indices:
                    valid_prompts = []
                    for p in next_prompts_list:
                        if p is None or p.numel() == 0:
                            valid_prompts.append(torch.tensor([pad_token_id], dtype=torch.long, device=gen_device))
                        else:
                            valid_prompts.append(p)

                    reversed_prompts = [p.flip(dims=(0,)) for p in valid_prompts]
                    padded_reversed = torch.nn.utils.rnn.pad_sequence(
                        reversed_prompts, batch_first=True, padding_value=pad_token_id
                    )
                    current_prompts = padded_reversed.flip(dims=(1,))

        return {
            'completions': trajectories,
            'dones': terminals,
            'generation_masks': generation_masks,
            'feedbacks': last_feedbacks
        }

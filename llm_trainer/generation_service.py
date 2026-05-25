from typing import Optional, List
import gc
import torch
import torch.distributed as dist
from llm_model import (
    LlmModel,
    ModelConfig
)

from .tools import TrainerTools
from .train_configs import GenerateConfig
from .generate_utils import batch_generate
from .utils import get_dtype, empty_cache
from .partition_utils import (
    get_full_state_dict_on_rank0,
    unwrap_model
)


class IndependentDeviceGenerationService:
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
            prompts: torch.Tensor,
            config: GenerateConfig,
            task_type: str,
            pixel_values: Optional[torch.Tensor] = None,
            tokens_per_image: Optional[int] = -1
    ) -> List[List[int]]:

        state_dict = get_full_state_dict_on_rank0(model)

        if TrainerTools().parallel.is_main_process:
            unwrapped_model = unwrap_model(model)
            prefix = 'policy_model.' if hasattr(unwrapped_model, 'policy_model') else ''
            clean_state_dict = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

            self.gen_model.load_state_dict(clean_state_dict)

            del state_dict
            del clean_state_dict
            gc.collect()
            empty_cache()

        local_prompts = prompts.cpu().tolist()
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

                max_new_tokens = max(config.max_seq_len - target_prompts_tensor.shape[1], 1)
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
                        temperature=config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                        repetition_penalty=config.repetition_penalty,
                        exclude_penalty_tokens=config.exclude_penalty_tokens,
                        suppress_tokens=config.suppress_tokens,
                        device=self.generation_device,
                        return_logits=False,
                    )

                    prompt_len = chunk_prompts.shape[1]
                    completions.extend(out_ids[:, prompt_len:].cpu().tolist())

                gathered_results[target_rank] = completions

        if self.world_size > 1:
            results_container = [gathered_results]
            dist.broadcast_object_list(results_container, src=0)
            gathered_results = results_container[0]

        return gathered_results[self.rank]

from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset

from .base_trainer import BaseTrainer
from .utils import pretrain_collate_fn
from .dataset import PretrainDataset
from .partition_utils import unwrap_model

from .train_configs import (
    TrainConfig,
    GenerationService
)
from .loss import (
    ChunkedLMLoss,
    LMLoss,
    KDLoss
)

class Trainer(BaseTrainer):
    """
    Trainer

    Args:
        train_config:
            - 全局训练配置，包含预训练配置 pretrain_config。

        eval_prompts:
            - 评估测试的提示词列表。
            - [num_eval_prompts] 长度的字符串列表。

        generation_service:
            - 自定义自回归生成服务。
    """
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            generation_service: Optional[GenerationService] = None,
    ):
        self.pretrain_config = train_config.pretrain_config
        self.chunked_cross_entropy_size = self.pretrain_config.chunked_cross_entropy_size

        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            generation_service=generation_service,
            gradient_accumulation_steps=self.pretrain_config.gradient_accumulation_steps,
            return_logits=self.chunked_cross_entropy_size is None
        )

        self.criterion, self.kd_loss = self._init_loss()

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": pretrain_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        block_size = self.train_config.dataset_block_size
        return PretrainDataset(file_path, block_size, block_size), file_path

    def _calc_attention_mask(self, inputs):
        return None

    def _init_loss(self) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
        if self.chunked_cross_entropy_size is not None:
            criterion = ChunkedLMLoss(chunk_size= self.chunked_cross_entropy_size)
        else:
            criterion = LMLoss()

        return criterion, KDLoss() if self.pretrain_config.kd_config else None

    def _calc_loss(self, inputs, attention_mask, result, labels) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        kd_config = self.pretrain_config.kd_config
        has_kd = (kd_config is not None and kd_config.kd_coef > 0.0)

        # calc loss
        if isinstance(self.criterion, ChunkedLMLoss):
            unwrapped = unwrap_model(self.train_model)
            lm_head = unwrapped.lm_head
            teacher_logits = None
            kd_coef = 0.0

            if has_kd:
                with torch.no_grad():
                    teacher_logits = kd_config.teacher_logits_provider(inputs, attention_mask)
                kd_coef = kd_config.kd_coef

            total_loss, ce_loss = self.criterion(
                hidden_states=result['hidden_states'],
                lm_head_weight=lm_head.weight,
                labels=labels,
                lm_head_bias=lm_head.bias,
                teacher_logits=teacher_logits,
                kd_coef=kd_coef
            )
            return total_loss, ce_loss

        ce_loss = self.criterion(result['logits'], labels)
        if not has_kd:
            return ce_loss, ce_loss

        with torch.no_grad():
            teacher_logits = kd_config.teacher_logits_provider(inputs, attention_mask)

        kd_loss = self.kd_loss(result['logits'], teacher_logits, labels)
        total_loss = (1.0 - kd_config.kd_coef) * ce_loss + kd_config.kd_coef * kd_loss
        return total_loss, ce_loss
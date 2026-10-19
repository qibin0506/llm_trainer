from typing import List, Tuple, Optional

import torch
from torch.utils.data import Dataset

from .base_trainer import BaseTrainer
from .utils import pretrain_collate_fn, TrainerTools
from .dataset import PretrainDataset
from .train_configs import TrainConfig, GenerationService
from .loss import LMLoss, KDLoss
from .partition_utils import add_fused_loss_head


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

    def _check_add_fused_head(self, model):
        if self.chunked_cross_entropy_size is not None:
            add_fused_loss_head(model, chunk_size=self.chunked_cross_entropy_size)

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": pretrain_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        block_size = self.train_config.dataset_block_size
        return PretrainDataset(file_path, block_size, block_size), file_path

    def _calc_attention_mask(self, inputs):
        return inputs != TrainerTools().tokenizer.pad

    def _init_loss(self) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
        return LMLoss(), KDLoss() if self.pretrain_config.kd_config else None

    def _get_teacher_logits_and_kd_coef(
            self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], float]:
        kd_config = self.pretrain_config.kd_config
        if kd_config is not None and kd_config.kd_coef > 0.0:
            with torch.no_grad():
                teacher_logits = kd_config.teacher_logits_provider(inputs, attention_mask)
            return teacher_logits, kd_config.kd_coef
        return None, 0.0

    def _calc_loss(
            self,
            inputs,
            attention_mask,
            result,
            labels,
            model: Optional[torch.nn.Module] = None,
            teacher_logits: Optional[torch.Tensor] = None,
            kd_coef: float = 0.0,
            loss_denom: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(result, dict) and "loss" in result:
            return result["loss"], result.get("ce_loss", result["loss"])

        ce_loss = self.criterion(result['logits'], labels, loss_denom=loss_denom)

        if teacher_logits is None and self.pretrain_config.kd_config is not None and self.pretrain_config.kd_config.kd_coef > 0.0:
            teacher_logits, kd_coef = self._get_teacher_logits_and_kd_coef(inputs, attention_mask)

        if teacher_logits is None or kd_coef <= 0.0:
            return ce_loss, ce_loss

        kd_loss = self.kd_loss(result['logits'], teacher_logits, labels, loss_denom=loss_denom)
        total_loss = (1.0 - kd_coef) * ce_loss + kd_coef * kd_loss
        return total_loss, ce_loss

from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset

from llm_model import (
    VLMConfig,
    LlmModel,
    VlmModel
)

from .base_trainer import BaseTrainer
from .dataset import SFTDataset
from .tools import TrainerTools

from .train_configs import (
    TrainConfig,
    GenerationService
)
from .loss import (
    LMLoss,
    KDLoss
)
from .utils import (
    get_sft_collate_fn,
    get_dtype
)


class SFTTrainer(BaseTrainer):
    """
    SFTTrainer

    Args:
        train_config:
            - 全局训练配置，若为多模态模型，model_config 需为 VLMConfig 类型。

        eval_prompts:
            - 评估阶段用于测试生成的文本提示词列表。
            - [num_eval_prompts] 长度的字符串列表。

        generation_service:
            - 外部自定义生成服务接口

        eval_image_tags:
            - 用于多模态评估时，与 eval_prompts 严格一一对应的图像 Tag 标识列表（用于通过图片加载器解析像素特征）。
            - 与 eval_prompts 长度相同的列表，即 [num_eval_prompts]。
    """
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            generation_service: Optional[GenerationService] = None,
            eval_image_tags: Optional[List[str]] = None
    ):
        self.sft_config = train_config.sft_config
        self.pixel_values_provider = self.sft_config.pixel_values_provider
        self.eval_image_tags = eval_image_tags

        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            generation_service=generation_service,
            gradient_accumulation_steps=self.sft_config.gradient_accumulation_steps
        )

        self.criterion, self.kd_loss = self._init_loss()

        if isinstance(train_config.model_config, VLMConfig):
            self.pixel_values_provider = self.sft_config.pixel_values_provider
        else:
            self.pixel_values_provider = None

    def _new_model(self, train_config: TrainConfig):
        if isinstance(train_config.model_config, VLMConfig):
            return VlmModel(train_config.model_config)
        else:
            return LlmModel(train_config.model_config)

    def _check_freeze_llm_model(self, model):
        # freeze llm model for vlm training
        if self.sft_config.freeze_llm_model:
            for name, param in model.named_parameters():
                if not any(sub_module in name for sub_module in ['multi_modal_projector']):
                    param.requires_grad = False

            # model.embed_tokens.eval()
            # model.layers.eval()
            # model.head_norm.eval()
            # model.lm_head.eval()

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        sft_collate_fn = get_sft_collate_fn(self.sft_config.mask_prompt)
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": sft_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _get_pixel_values(self, batch_data):
        if self.pixel_values_provider and 'image_tags' in batch_data:
            valid_tags = [tag for tag in batch_data['image_tags'] if tag]
            if len(valid_tags) > 0:
                pixel_values = self.pixel_values_provider(valid_tags)
                if pixel_values is not None:
                    return pixel_values.to(TrainerTools().parallel.device, get_dtype())

        return None

    def _get_eval_pixel_values_and_tokens_count(self, eval_idx):
        if not self.eval_image_tags:
            return None, None

        eval_image_tag = self.eval_image_tags[eval_idx]
        if isinstance(self.train_config.model_config, VLMConfig) and self.pixel_values_provider and eval_image_tag:
            return self.pixel_values_provider([eval_image_tag]), self.train_config.model_config.tokens_per_image

        return None, None

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        block_size = self.train_config.dataset_block_size

        image_tag_file_path = None
        tokens_per_image = -1

        if isinstance(self.train_config.model_config, VLMConfig):
            if self.sft_config.image_tags_file_dataset:
                image_tag_file_path = self.sft_config.image_tags_file_dataset[file_idx]

            if self.train_config.model_config.tokens_per_image:
                tokens_per_image = self.train_config.model_config.tokens_per_image

        return SFTDataset(file_path, block_size, image_tag_file_path, tokens_per_image), file_path

    def _init_loss(self) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
        return LMLoss(), KDLoss() if self.sft_config.kd_config else None

    def _calc_loss(self, inputs, attention_mask, logits, labels) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # calc loss
        ce_loss = self.criterion(logits, labels)
        if not self.kd_loss or self.sft_config.kd_config.kd_coef == 0.0:
            # 不用计算kd_loss
            return ce_loss, ce_loss

        teacher_logits = self.sft_config.kd_config.teacher_logits_provider(inputs, attention_mask)
        loss = self.kd_loss(logits, teacher_logits, labels)

        return (1 - self.sft_config.kd_config.kd_coef) * ce_loss + self.sft_config.kd_config.kd_coef * loss, ce_loss

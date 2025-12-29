from typing import Optional, Tuple, List
from torch.utils.data import Dataset

from llm_model import (
    VLMConfig,
    LlmModel,
    VlmModel
)

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig
from .dataset import SFTDataset
from .utils import get_sft_collate_fn
from .tools import TrainerTools


class SFTTrainer(BaseTrainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            eval_image_tags: Optional[List[str]] = None
    ):
        self.sft_config = train_config.sft_config
        self.pixel_values_provider = self.sft_config.pixel_values_provider
        self.eval_image_tags = eval_image_tags

        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            kd_config=self.sft_config.kd_config,
            gradient_accumulation_steps=self.sft_config.gradient_accumulation_steps
        )

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
            image_tags = batch_data['image_tags']
            return self.pixel_values_provider(image_tags).to(TrainerTools().parallel.device)

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
        max_seq_len = self.train_config.max_seq_len

        image_tag_file_path = None
        tokens_per_image = -1

        if isinstance(self.train_config.model_config, VLMConfig):
            if self.sft_config.image_tags_file_dataset:
                image_tag_file_path = self.sft_config.image_tags_file_dataset[file_idx]

            if self.train_config.model_config.tokens_per_image:
                tokens_per_image = self.train_config.model_config.tokens_per_image

        return SFTDataset(file_path, max_seq_len, image_tag_file_path, tokens_per_image), file_path

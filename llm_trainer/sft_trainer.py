from typing import Optional, Tuple, List

from torch.utils.data import Dataset

from .trainer import Trainer
from .train_configs import TrainConfig, VLMConfig
from .dataset import LineByLineTextDataset
from .utils import get_sft_collate_fn


class SFTTrainer(Trainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            eval_image_tags: Optional[List[str]] = None
    ):
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            eval_image_tags=eval_image_tags
        )
        self.packed_sequences = False

    def _convert_train_args(self) -> Tuple[dict, dict, dict, bool]:
        sft_collate_fn = get_sft_collate_fn(self.train_config.mask_prompt)
        parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": sft_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        max_position_embeddings = self.train_config.model_config.max_position_embeddings

        image_tag_file_path = None
        tokens_per_image = -1

        if isinstance(self.train_config.model_config, VLMConfig):
            if self.train_config.image_tags_file_dataset:
                image_tag_file_path = self.train_config.image_tags_file_dataset[file_idx]

            if self.train_config.model_config.tokens_per_image:
                tokens_per_image = self.train_config.model_config.tokens_per_image

        return LineByLineTextDataset(file_path, max_position_embeddings, image_tag_file_path, tokens_per_image), file_path

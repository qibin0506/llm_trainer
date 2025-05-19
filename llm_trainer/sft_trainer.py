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
            eval_image_tags: Optional[List[int]] = None
    ):
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            eval_image_tags=eval_image_tags
        )

    def _convert_train_args(self) -> Tuple[dict, dict, dict, bool]:
        sft_collate_fn = get_sft_collate_fn(self.train_config.mask_prompt)
        parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": sft_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim

    def _create_dataset(self, file_path) -> Dataset:
        max_position_embeddings = self.train_config.model_config.max_position_embeddings
        if isinstance(self.train_config.model_config, VLMConfig):
            tokens_per_image = self.train_config.model_config.tokens_per_image
        else:
            tokens_per_image = -1

        return LineByLineTextDataset(file_path, max_position_embeddings, tokens_per_image)

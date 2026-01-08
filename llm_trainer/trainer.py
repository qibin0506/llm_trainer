from typing import List, Tuple

from torch.utils.data import Dataset

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig
from .utils import pretrain_collate_fn
from .dataset import PretrainDataset


class Trainer(BaseTrainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
    ):
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            kd_config=train_config.pretrain_config.kd_config,
            gradient_accumulation_steps=train_config.pretrain_config.gradient_accumulation_steps
        )

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": pretrain_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        block_size = self.train_config.dataset_block_size
        return PretrainDataset(file_path, block_size, block_size), file_path
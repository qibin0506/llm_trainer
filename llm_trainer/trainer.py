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
        max_seq_len = self.train_config.max_seq_len
        return PretrainDataset(file_path, max_seq_len, max_seq_len), file_path
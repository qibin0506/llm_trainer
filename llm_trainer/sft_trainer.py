from typing import Tuple, List

from torch.utils.data import Dataset

from .trainer import Trainer
from .train_configs import TrainConfig
from .dataset import LineByLineTextDataset
from .utils import sft_collate_fn

class SFTTrainer(Trainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str]
    ):
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts
        )

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": sft_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_path) -> Dataset:
        max_position_embeddings = self.train_config.llama_config.max_position_embeddings
        return LineByLineTextDataset(file_path, max_position_embeddings)

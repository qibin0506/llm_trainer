from .trainer import Trainer
from .train_args import TrainArgs

class SFTTrainer(Trainer):
    def __init__(
            self,
            train_args: TrainArgs,
            prompt_on_batch: str,
            prompt_on_epoch: str,
    ):
        super().__init__(
            train_args=train_args,
            prompt_on_batch=prompt_on_batch,
            prompt_on_epoch=prompt_on_epoch
        )

    def _is_sft(self) -> bool:
        return True
from typing import Optional
import torch
from torch import nn

from pytorch.llm.llm_trainer.parallel import Parallel

class NoneParallel(Parallel):
    def __init__(self):
        super().__init__(user_parallel=False)

    def process_model(
            self,
            model: nn.Module,
            ckpt_path: str,
            kwargs: Optional[dict] = None
    ) -> nn.Module:
        model.to(self.device)

        # 先load state, 再compile，最后DDP
        self._load_ckpt(model, ckpt_path)

        if self._use_compile:
            model = torch.compile(model)

        self.raw_model = model
        self.model = model

        return self.model



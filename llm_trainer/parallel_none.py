from typing import Optional
import torch
from torch import nn

from .parallel import Parallel

class NoneParallel(Parallel):
    def __init__(self):
        super().__init__(user_parallel=False)

    def process_model(
            self,
            model: nn.Module,
            kwargs: Optional[dict] = None
    ) -> nn.Module:
        model.to(self.device)

        if self._use_compile:
            model = torch.compile(model)

        self.raw_model = model
        self.model = model

        return self.model



from typing import Optional, Tuple
import torch
from torch import nn

from .parallel import Parallel

class NoneParallel(Parallel):
    def __init__(self):
        super().__init__(use_parallel=False)

    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None,
            save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        model.to(self.device)

        if self._use_compile:
            model = torch.compile(model)

        if save_instance:
            self.raw_model = model
            self.model = model

        return model, optimizer




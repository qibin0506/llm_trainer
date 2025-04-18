from typing import Optional, Tuple
import torch
from torch import nn
import deepspeed
from .parallel import Parallel

class DsParallel(Parallel):
    def __init__(self):
        deepspeed.init_distributed(dist_backend='nccl')
        super().__init__(init_process_group=False)

    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        :param model:
        :param optimizer:
        :param kwargs:
            参考deepspeed配置
        :return:
        """
        self.raw_model = model

        model, optim, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            dist_init_required=False,
            config_params=kwargs
        )

        self.model = model
        return model, optim

    def synchronize(self):
        pass

    def destroy(self):
        pass




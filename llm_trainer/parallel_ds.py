from typing import Optional, Tuple
import torch
from torch import nn
import deepspeed
from .parallel import Parallel

class FsdpParallel(Parallel):
    def __init__(self):
        super().__init__()

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
            "wrap_policy_num_params" int size_based_auto_wrap_policy的最小参数量
            "cpu_offload" bool 是否使用cpu卸载
            "offload_params" bool 是否卸载参数，在cpu_offload为True时生效
        :return:
        """
        deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            dist_init_required=True
        )




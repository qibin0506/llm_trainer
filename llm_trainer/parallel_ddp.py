from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .parallel import Parallel


# python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 gpt.py
# torchrun --standalone --nproc_per_node=gpu pretrain.py
#    --standalone 代表单机运行
#    --nproc_per_node=gpu 代表使用所有可用GPU, 等于号后也可写gpu数量n, 这样会使用前n个GPU


class DdpParallel(Parallel):
    def __init__(self):
        super().__init__()

    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        model.to(self.device)

        if self._use_compile:
            model = torch.compile(model)

        if self._use_parallel:
            # self.model = DDP(module=model, broadcast_buffers=False, find_unused_parameters=True)
            self.model = DDP(module=model, device_ids=[self._local_rank], output_device=self._local_rank)
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

        return self.model, optimizer

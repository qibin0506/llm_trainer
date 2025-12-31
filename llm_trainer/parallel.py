import os
from typing import Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import deepspeed
except: ...

from .log import Logger


class Parallel(ABC):
    def __init__(
            self,
            _init_process_group: bool = True,
            _use_parallel: bool = True
    ):
        self._initialize(_init_process_group, _use_parallel)

    def _initialize(
            self,
            _init_process_group: bool,
            _use_parallel: bool
    ):
        self._global_rank: int = int(os.environ.get('RANK', -1))
        self._local_rank: int = int(os.environ.get('LOCAL_RANK', -1))
        self._use_parallel: bool = _use_parallel and self._global_rank != -1

        self._sampler: Optional[DistributedSampler] = None
        self.model: Optional[nn.Module] = None

        try:
            torch.set_float32_matmul_precision('high')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except:
            pass

        if self._use_parallel:
            if _init_process_group:
                dist.init_process_group(backend='nccl')

            self.device: str = f'cuda:{self._local_rank}'
            self.device_type: str = 'cuda'
            torch.cuda.set_device(self.device)

            Logger.std_log(f'global_rank={self._global_rank}, local_rank={self._local_rank}, world_size={self.world_size}')
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

            self.device: str = device
            self.device_type: str = device

    @abstractmethod
    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None,
            save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]: ...

    def process_dataloader(
            self,
            dataset: Dataset,
            data_loader_kwargs: dict,
            sampler_kwargs: Optional[dict]=None
    ) -> DataLoader:
        """
        :param dataset:
        :param data_loader_kwargs
                "batch_size" int,
                "pin_memory" bool,
                "collate_fn" collate_fn,
                "num_workers" int
                "shuffle" bool
                "drop_last" bool
        :param sampler_kwargs:
                "shuffle" bool
                "drop_last" bool
        :return:
        """

        if self._use_parallel:
            self._sampler = DistributedSampler(dataset=dataset, **sampler_kwargs)
            return DataLoader(dataset=dataset, sampler=self._sampler, **data_loader_kwargs)

        return DataLoader(dataset=dataset, **data_loader_kwargs)

    def on_epoch_start(self, epoch):
        if self._sampler:
            self._sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch): ...

    def synchronize(self):
        if self._use_parallel:
            torch.cuda.synchronize(device=self.device)

    def destroy(self):
        if self._use_parallel:
            dist.destroy_process_group()

    @property
    def parallel_train(self) -> bool:
        return self._use_parallel

    @property
    def is_main_process(self) -> bool:
        if self._use_parallel:
            return self._global_rank == 0

        return True

    @property
    def world_size(self) -> int:
        if self._use_parallel:
            return dist.get_world_size()
        return 1

    def wait(self, msg=None):
        if self.world_size == 1:
            return

        msg = f' for {msg}' if msg else ''
        Logger.std_log(f'wait at {self.device}{msg}')
        dist.barrier()
        Logger.std_log(f'continue at {self.device}{msg}')


class DsParallel(Parallel):
    def __init__(self):
        deepspeed.init_distributed(dist_backend='nccl')
        super().__init__(_init_process_group=False)

    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None,
            save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
            :param model:
            :param optimizer:
            :param kwargs:
                参考deepspeed配置
            :param save_instance
            :return:
        """
        model, optim, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            dist_init_required=False,
            config_params=kwargs
        )

        if save_instance:
            self.model = model

        return model, optim

    def synchronize(self): ...

    def destroy(self): ...


class DdpParallel(Parallel):
    def __init__(self):
        super().__init__()

    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None,
            save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        model.to(self.device)

        if self._use_parallel:
            # self.model = DDP(module=model, broadcast_buffers=False, find_unused_parameters=True)
            model = DDP(module=model, device_ids=[self._local_rank], output_device=self._local_rank)
        else:
            model = model

        if save_instance:
            self.model = model

        return model, optimizer


class NoneParallel(Parallel):
    def __init__(self):
        super().__init__(_use_parallel=False)

    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None,
            save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        model.to(self.device)

        if save_instance:
            self.model = model

        return model, optimizer
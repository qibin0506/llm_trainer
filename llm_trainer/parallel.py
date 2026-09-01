import os
from typing import Optional, Tuple
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import deepspeed
except ImportError:
    deepspeed = None

from .log import Logger


def _detect_device_and_backend() -> Tuple[str, str]:
    user_backend = os.environ.get('USER_BACKEND', '').strip()
    if hasattr(torch, 'mlu') and torch.mlu.is_available():
        return 'mlu', user_backend or ('cncl' if hasattr(dist, 'is_cncl_available') and dist.is_cncl_available() else 'gloo')
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        return 'npu', user_backend or ('hccl' if hasattr(dist, 'is_hccl_available') and dist.is_hccl_available() else 'gloo')
    elif torch.cuda.is_available():
        return 'cuda', user_backend or ('nccl' if dist.is_nccl_available() else 'gloo')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps', user_backend or 'gloo'
    else:
        return 'cpu', user_backend or 'gloo'


class Parallel(ABC):
    def __init__(self, init_process_group: bool = True, use_parallel: bool = True):
        self._global_rank: int = int(os.environ.get('RANK', -1))
        self._local_rank: int = int(os.environ.get('LOCAL_RANK', -1))
        self._world_size: int = int(os.environ.get('WORLD_SIZE', 1))

        self.device_type, self.dist_backend = _detect_device_and_backend()
        self._use_parallel: bool = use_parallel and (self._global_rank != -1)
        self._sampler: Optional[DistributedSampler] = None
        self.model: Optional[nn.Module] = None

        self._setup_hardware()
        if self._use_parallel and init_process_group and not dist.is_initialized():
            dist.init_process_group(backend=self.dist_backend)

        Logger.std_log(
            f'backend={self.dist_backend}, global_rank={self.global_rank}, '
            f'local_rank={self.local_rank}, world_size={self.world_size}, '
            f'device_type={self.device_type}, device={self.device}'
        )

    def _setup_hardware(self):
        if self.device_type == 'cuda':
            try:
                torch.set_float32_matmul_precision('high')
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except: ...

        if self._use_parallel and self._local_rank != -1:
            self.device = f"{self.device_type}:{self._local_rank}"
            if self.device_type == 'cuda':
                torch.cuda.set_device(self._local_rank)
            elif self.device_type == 'npu':
                torch.npu.set_device(self._local_rank)
            elif self.device_type == 'mlu':
                torch.mlu.set_device(self._local_rank)
        else:
            self.device = self.device_type

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
            sampler_kwargs: Optional[dict] = None
    ) -> DataLoader:
        if self._use_parallel:
            sampler_kwargs = sampler_kwargs or {}
            self._sampler = DistributedSampler(dataset=dataset, **sampler_kwargs)
            data_loader_kwargs = {k: v for k, v in data_loader_kwargs.items() if k != "shuffle"}
            return DataLoader(dataset=dataset, sampler=self._sampler, **data_loader_kwargs)

        return DataLoader(dataset=dataset, **data_loader_kwargs)

    def on_epoch_start(self, epoch):
        if self._sampler:
            self._sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch): ...

    def synchronize(self):
        if self._use_parallel:
            if self.device_type == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif self.device_type == 'npu' and hasattr(torch, 'npu'):
                torch.npu.synchronize()
            elif self.device_type == 'mlu' and hasattr(torch, 'mlu'):
                torch.mlu.synchronize()
            elif self.device_type == 'mps' and hasattr(torch, 'mps'):
                torch.mps.synchronize()

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
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def global_rank(self) -> int:
        return self._global_rank

    @property
    def world_size(self) -> int:
        if self._use_parallel:
            if dist.is_initialized():
                return dist.get_world_size()
            return self._world_size
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
        if not deepspeed:
            raise ImportError("DeepSpeed not installed.")

        super().__init__(init_process_group=False)
        if self._use_parallel and not dist.is_initialized():
            deepspeed.init_distributed(dist_backend=self.dist_backend)

    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None,
            save_instance: bool = True
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:

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


class NoneParallel(Parallel):
    def __init__(self):
        super().__init__(init_process_group=False, use_parallel=False)

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
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


def _get_optimal_backend():
    user_backend = os.environ.get('USER_BACKEND', '')
    if user_backend != '':
        return user_backend

    if hasattr(torch, 'mlu') and torch.mlu.is_available() and hasattr(dist, 'is_cncl_available') and dist.is_cncl_available():
        return 'cncl'

    if hasattr(dist, 'is_hccl_available') and dist.is_hccl_available():
        return 'hccl'

    if torch.cuda.is_available() and dist.is_nccl_available():
        return 'nccl'

    return 'gloo'


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
        self.dist_backend = _get_optimal_backend()

        if self._global_rank == -1:
            _use_parallel = False

        self._use_parallel: bool = _use_parallel and self._global_rank != -1

        self._sampler: Optional[DistributedSampler] = None
        self.model: Optional[nn.Module] = None

        try:
            if torch.cuda.is_available():
                torch.set_float32_matmul_precision('high')
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except: ...

        if self._use_parallel:
            if self.dist_backend == 'nccl':
                self.device_type = 'cuda'
                self.device = f'cuda:{self._local_rank}'
                torch.cuda.set_device(self.device)
            elif self.dist_backend == 'cncl':
                self.device_type = 'mlu'
                self.device = f'mlu:{self._local_rank}'
                torch.mlu.set_device(self.device)
            elif self.dist_backend == 'hccl':
                self.device_type = 'npu'
                self.device = f'npu:{self._local_rank}'
                torch.npu.set_device(self.device)
            else:
                # gloo
                if hasattr(torch, 'mlu') and torch.mlu.is_available():
                    self.device_type = 'mlu'
                    self.device = f'mlu:{self._local_rank}'
                    torch.mlu.set_device(self.device)
                elif hasattr(torch, 'npu') and torch.npu.is_available():
                    self.device_type = 'npu'
                    self.device = f'npu:{self._local_rank}'
                    torch.npu.set_device(self.device)
                elif torch.cuda.is_available():
                    self.device_type = 'cuda'
                    self.device = f'cuda:{self._local_rank}'
                    torch.cuda.set_device(self.device)
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device_type = 'mps'
                    self.device = 'mps'
                else:
                    self.device_type = 'cpu'
                    self.device = 'cpu'

            if _init_process_group:
                dist.init_process_group(backend=self.dist_backend)
        else:
            if hasattr(torch, 'mlu') and torch.mlu.is_available():
                self.device_type = 'mlu'
                self.device = "mlu"
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                self.device_type = 'npu'
                self.device = "npu"
            elif torch.cuda.is_available():
                self.device_type = 'cuda'
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device_type = 'mps'
                self.device = "mps"
            else:
                self.device_type = 'cpu'
                self.device = "cpu"

        Logger.std_log(f'backend={self.dist_backend}, global_rank={self._global_rank}, local_rank={self._local_rank}, world_size={self.world_size}, device_type={self.device_type}, device={self.device}')

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
            return DataLoader(dataset=dataset, sampler=self._sampler, **data_loader_kwargs)

        return DataLoader(dataset=dataset, **data_loader_kwargs)

    def on_epoch_start(self, epoch):
        if self._sampler:
            self._sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch): ...

    def synchronize(self):
        if self._use_parallel:
            if self.device_type == 'cuda':
                torch.cuda.synchronize(device=self.device)
            elif self.device_type == 'npu':
                torch.npu.synchronize()
            elif self.device_type == 'mlu':
                torch.mlu.synchronize()
            elif self.device_type == 'mps':
                torch.mps.synchronize()
            else: ...

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
            return int(os.environ.get('WORLD_SIZE', 1))
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
        super().__init__()

        if deepspeed:
            deepspeed.init_distributed(dist_backend=self.dist_backend)
        else:
            raise ImportError("DeepSpeed not installed.")

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
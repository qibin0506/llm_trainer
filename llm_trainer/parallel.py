import os
from typing import Optional

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from .utils import log


class Parallel:
    def __init__(
            self,
            user_parallel: bool = True,
            init_process_group: bool = True
    ):
        self._use_compile: bool = False
        self._global_rank: int = int(os.environ.get('RANK', -1))
        self._local_rank: int = int(os.environ.get('LOCAL_RANK', -1))
        self._use_parallel: bool = user_parallel and self._global_rank != -1
        self._word_size: int = 1

        self._sampler:Optional[DistributedSampler] = None

        self.model: Optional[nn.Module] = None
        self.raw_model: Optional[nn.Module] = None

        if self._use_compile:
            torch.set_float32_matmul_precision('high')

        if self._use_parallel:
            if init_process_group:
                dist.init_process_group(backend='nccl')

            self.device: str = f'cuda:{self._local_rank}'
            self.device_type: str = 'cuda'
            self._word_size = dist.get_world_size()

            torch.cuda.set_device(self.device)

            log(f'global_rank:{self._global_rank},local_rank:{self._local_rank}, world_size:{self._word_size}')
        else:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"

            self.device: str = device
            self.device_type: str = device

    def process_model(
            self,
            model: nn.Module,
            kwargs: Optional[dict] = None
    ) -> nn.Module:
        raise NotImplementedError()

    # def _load_ckpt(self, model: nn.Module, ckpt_path: str):
    #     # 先load state, 再compile，最后DDP
    #     if self._use_parallel:
    #         ddp_init_path = f'{ckpt_path}_parallel_init.pth'
    #         if not os.path.exists(ckpt_path):
    #             if self.is_main_process:
    #                 ckpt = {'model': model.state_dict()}
    #                 torch.save(ckpt, ddp_init_path)
    #
    #             dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    #             ckpt = torch.load(ddp_init_path, map_location=self.device)
    #             model.load_state_dict(ckpt['model'])
    #             print(f'load init ckpt for {self.device}')
    #         else:
    #             ckpt = torch.load(ckpt_path, map_location=self.device)
    #             model.load_state_dict(ckpt['model'])
    #     else:
    #         if os.path.exists(ckpt_path):
    #             ckpt = torch.load(ckpt_path, map_location=self.device)
    #             model.load_state_dict(ckpt['model'])


    def create_dataloader(
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
        if self._sampler is not None:
            self._sampler.set_epoch(epoch)

    def on_epoch_end(self, epoch):
        pass

    def synchronize(self):
        if self._use_parallel:
            torch.cuda.synchronize(device=self.device)

    def destroy(self):
        if self._use_parallel:
            dist.destroy_process_group()

    def reduce_loss(self, avg_loss: torch.Tensor, loss: torch.Tensor, batch) -> torch.Tensor:
        if self._use_parallel:
            world_size = dist.get_world_size()
            if world_size < 2:
                return loss.detach()

            torch.distributed.all_reduce(loss)
            # 整个训练过程的滑动损失均值=在历史平均损失的基础上，加上最新损失再求平均
            avg_loss = (avg_loss * batch + loss.detach()) / (batch + 1)
            return avg_loss

        return loss.detach()

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
        return self._word_size

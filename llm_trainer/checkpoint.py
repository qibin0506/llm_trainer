import os
from typing import Optional, Union
import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .parallel import DsParallel
from .scheduler import LRScheduler
from .tools import TrainerTools

DEFAULT_CHECKPOINT_NAME = "checkpoint.pth"

def save_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        extra_module: Optional[nn.Module] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import save_ds_checkpoint
        save_ds_checkpoint(model, extra_module=extra_module)
    else:
        if TrainerTools().parallel.is_main_process:
            checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)

            raw_model = model if not isinstance(model, DDP) else model.module
            ckpt = {'model_state_dict': raw_model.state_dict()}

            if optimizer:
                ckpt.update({'optim_state_dict': optimizer.state_dict()})

            if extra_module:
                ckpt.update({'extra_module_state_dict': extra_module.state_dict()})

            torch.save(ckpt, checkpoint_name)


def load_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[Union[torch.device, str]] = None,
        load_module_only: bool = False,
        extra_module: Optional[nn.Module] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import load_ds_checkpoint
        load_ds_checkpoint(model, load_module_only=load_module_only, extra_module=extra_module)
    else:
        checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)

        if os.path.exists(checkpoint_name):
            state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
            raw_model = model.module if isinstance(model, DDP) else model
            raw_model.load_state_dict(state_dict['model_state_dict'])

            if optimizer and 'optim_state_dict' in state_dict:
                optimizer.load_state_dict(state_dict['optim_state_dict'])

            if extra_module and 'extra_module_state_dict' in state_dict:
                extra_module.load_state_dict(state_dict['extra_module_state_dict'])


def save_steps(
    epoch: int = 0,
    file_idx: int = 0,
    batch_idx: int = 0,
    lr_scheduler: Optional[LRScheduler] = None
):
    # 暂时只保存主进程的
    if TrainerTools().parallel.is_main_process:
        steps_checkpoint_name = f"{os.environ.get('LOG_DIR', './')}steps.pt"
        ckpt = {
            'epoch': epoch,
            'file_idx': file_idx,
            'batch_idx': batch_idx,
            'cpu_rng_state': torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            ckpt['cuda_rng_state'] = torch.cuda.get_rng_state()

        if lr_scheduler:
            ckpt.update(lr_scheduler.get_ckpt_dict())

        torch.save(ckpt, steps_checkpoint_name)


def load_steps() -> Optional[dict]:
    steps_dict = None

    if TrainerTools().parallel.is_main_process:
        steps_checkpoint_name = f"{os.environ.get('LOG_DIR', './')}steps.pt"
        if os.path.exists(steps_checkpoint_name):
            try:
                steps_dict = torch.load(steps_checkpoint_name, weights_only=True)
            except:
                steps_dict = None

    if TrainerTools().parallel.world_size > 1:
        object_list = [steps_dict]
        dist.broadcast_object_list(object_list, src=0)
        steps_dict = object_list[0]
        TrainerTools().parallel.wait('broadcast steps_dict')

    if steps_dict:
        if 'cpu_rng_state' in steps_dict:
            torch.set_rng_state(steps_dict['cpu_rng_state'])
        if 'cuda_rng_state' in steps_dict and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state(steps_dict['cuda_rng_state'])
            except: ...

    return steps_dict
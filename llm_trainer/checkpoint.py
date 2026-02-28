import os
from typing import Optional, Union
import torch
from torch import nn
from torch.optim import Optimizer
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
            ckpt = {'model_state_dict': model.state_dict()}

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
            model.load_state_dict(state_dict['model_state_dict'])

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
    if TrainerTools().parallel.is_main_process:
        steps_checkpoint_name = f"{os.environ.get('LOG_DIR', './')}steps.pt"
        ckpt = {
            'epoch': epoch,
            'file_idx': file_idx,
            'batch_idx': batch_idx,
        }

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

    return steps_dict
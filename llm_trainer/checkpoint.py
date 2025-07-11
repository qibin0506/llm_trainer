import os
from typing import Optional, Union, Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from .parallel_ds import DsParallel
from .scheduler import LRScheduler
from .tools import TrainerTools

DEFAULT_CHECKPOINT_NAME = "checkpoint.pth"

def save_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        suffix: Optional[str] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import save_ds_checkpoint
        save_ds_checkpoint(model, suffix)
    else:
        if TrainerTools().parallel.is_main_process:
            checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
            if suffix:
                checkpoint_name = f"{checkpoint_name}_{suffix}"

            raw_model = model if not isinstance(model, DDP) else model.module
            ckpt = {'model_state_dict': raw_model.state_dict()}

            if optimizer:
                ckpt.update({'optim_state_dict': optimizer.state_dict()})

            torch.save(ckpt, checkpoint_name)


def load_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[Union[torch.device, str]] = None,
        load_module_only: bool = False,
        suffix: Optional[str] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import load_ds_checkpoint
        load_ds_checkpoint(model, load_module_only=load_module_only, suffix=suffix)
    else:
        checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
        if suffix:
            checkpoint_name = f"{checkpoint_name}_{suffix}"

        state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
        raw_model = model.module if isinstance(model, DDP) else model
        raw_model.load_state_dict(state_dict['model_state_dict'])

        if optimizer:
            optimizer.load_state_dict(state_dict['optim_state_dict'])


def load_checkpoint_for_eval(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        suffix: Optional[str] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import load_ds_checkpoint_for_eval
        load_ds_checkpoint_for_eval(model)
    else:
        load_checkpoint(model, None, device, suffix=suffix)


def save_steps(global_steps: int, lr_scheduler: Optional[LRScheduler] = None):
    # 暂时只保存主进程的
    if TrainerTools().parallel.is_main_process:
        steps_checkpoint_name = f"{os.environ.get('LOG_DIR', './')}steps.pt"
        ckpt = {'global_steps': global_steps, 'lr_steps': lr_scheduler.cur_steps}
        torch.save(ckpt, steps_checkpoint_name)


def load_steps(
        default_global_steps: int = 0,
        default_lr_steps: int = 0
) -> Tuple[Optional[int], Optional[int]]:
    steps_checkpoint_name = f"{os.environ.get('LOG_DIR', './')}steps.pt"
    if os.path.exists(steps_checkpoint_name):
        ckpt = torch.load(steps_checkpoint_name, weights_only=True)
        return ckpt['global_steps'], ckpt['lr_steps']

    return default_global_steps, default_lr_steps

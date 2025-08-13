import os
from typing import Optional, Union
import shutil
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
        optimizer: Optional[Optimizer] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import save_ds_checkpoint
        save_ds_checkpoint(model)
    else:
        if TrainerTools().parallel.is_main_process:
            checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)

            raw_model = model if not isinstance(model, DDP) else model.module
            ckpt = {'model_state_dict': raw_model.state_dict()}

            if optimizer:
                ckpt.update({'optim_state_dict': optimizer.state_dict()})

            torch.save(ckpt, checkpoint_name)


def save_best_checkpoint(
        current_loss: float,
        last_best_checkpoint_loss: Optional[float] = None
) -> bool:
    # 指定不保存最佳checkpoint
    if os.environ.get('SAVE_BEST_CHECKPOINT', '1') != '1':
        return False

    need_replace = not last_best_checkpoint_loss or current_loss <= last_best_checkpoint_loss
    if need_replace and TrainerTools().parallel.is_main_process:
        try:
            if isinstance(TrainerTools().parallel, DsParallel):
                checkpoint_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')

                if checkpoint_dir.endswith('/'):
                    best_checkpoint_dir = f'{checkpoint_dir[:-1]}_best'
                else:
                    best_checkpoint_dir = f'{checkpoint_dir}_best'

                if not os.path.exists(best_checkpoint_dir):
                    os.makedirs(best_checkpoint_dir)

                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(best_checkpoint_dir)
                    shutil.copytree(checkpoint_dir, best_checkpoint_dir)
            else:
                checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
                best_checkpoint_name = f'{checkpoint_name}_best'

                if os.path.exists(checkpoint_name):
                    if os.path.exists(best_checkpoint_name):
                        os.remove(best_checkpoint_name)

                    shutil.copy2(checkpoint_name, best_checkpoint_name)
        except: pass

    TrainerTools().parallel.wait('save best checkpoint')
    return need_replace


def load_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[Union[torch.device, str]] = None,
        load_module_only: bool = False
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import load_ds_checkpoint
        load_ds_checkpoint(model, load_module_only=load_module_only)
    else:
        checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)

        if os.path.exists(checkpoint_name):
            state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
            raw_model = model.module if isinstance(model, DDP) else model
            raw_model.load_state_dict(state_dict['model_state_dict'])

            if optimizer:
                optimizer.load_state_dict(state_dict['optim_state_dict'])


def load_checkpoint_for_eval(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import load_ds_checkpoint_for_eval
        load_ds_checkpoint_for_eval(model)
    else:
        load_checkpoint(model, None, device)


def save_steps(global_steps: int, lr_scheduler: Optional[LRScheduler] = None):
    # 暂时只保存主进程的
    if TrainerTools().parallel.is_main_process:
        steps_checkpoint_name = f"{os.environ.get('LOG_DIR', './')}steps.pt"
        ckpt = {'global_steps': global_steps}
        ckpt.update(lr_scheduler.get_ckpt_dict())
        torch.save(ckpt, steps_checkpoint_name)


def load_steps() -> Optional[dict]:
    steps_checkpoint_name = f"{os.environ.get('LOG_DIR', './')}steps.pt"
    if os.path.exists(steps_checkpoint_name):
        return torch.load(steps_checkpoint_name, weights_only=True)

    return None

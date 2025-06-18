import os
from typing import Optional, Union, Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from .parallel_ds import DsParallel
from .parallel_fsdp import FsdpParallel
from .parallel_ddp import DdpParallel
from .scheduler import LRScheduler
from .tools import TrainerTools

try:
    from .dcp import save_dcp, load_dcp, convert_dcp_to_pth
except:
    os.environ['ENABLE_DCP'] = "0"

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

DEFAULT_CHECKPOINT_NAME = "checkpoint.pth"


def _can_use_dcp(model: nn.Module) -> bool:
    if os.environ.get('ENABLE_DCP', '1') != '1':
        return False

    # 如果是fsdp或者ddp，才能使用dcp保存
    if (isinstance(TrainerTools().parallel, FsdpParallel)
            or isinstance(TrainerTools().parallel, DdpParallel)):
        return True

    return False


def save_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        suffix: Optional[str] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import save_ds_checkpoint
        save_ds_checkpoint(model, suffix)
    elif _can_use_dcp(model):
        save_dcp(model, optimizer, suffix)
    elif isinstance(model, FSDP):
        from .fsdp_checkpoint import save_fsdp_checkpoint
        save_fsdp_checkpoint(model, optimizer, suffix)
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
    elif _can_use_dcp(model):
        load_dcp(model, optimizer, suffix)
    elif isinstance(model, FSDP):
        from .fsdp_checkpoint import load_fsdp_checkpoint
        load_fsdp_checkpoint(model, optimizer, device, suffix)
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
    elif _can_use_dcp(model):
        checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)

        # load_dcp方式在cpu上会报错，所以改为先将ckpt转换为pth，然后再加载pth
        # load_dcp(model, optimizer)
        pth_name = os.environ.get('EVAL_CHECKPOINT_NAME', checkpoint_name)
        if suffix:
            pth_name = f'{pth_name}_{suffix}'

        convert_dcp_to_pth(pth_name)

        if os.path.exists(pth_name):
            ckpt = torch.load(pth_name, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['app']['model_state_dict'])
            # 使用完删除
            os.remove(pth_name)
    else:
        load_checkpoint(model, None, device, suffix=suffix)


def copy_model_params(
        _from: nn.Module,
        _to: Optional[nn.Module]
):
    """
        必须在所有rank上调用，非rank0, _to可以设置为None
    """

    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import get_ds_model_params
        state_dict = get_ds_model_params(_from)
    elif isinstance(TrainerTools().parallel, FsdpParallel):
        from .fsdp_checkpoint import get_fsdp_model_params
        state_dict = get_fsdp_model_params(_from)
    elif isinstance(_from, DDP):
        state_dict = _from.module.state_dict()
    else:
        state_dict = _from.state_dict()

    if _to and state_dict:
        _to.load_state_dict(state_dict)


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

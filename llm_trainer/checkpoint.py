import os
from typing import Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
import torch.distributed as dist

from .parallel import DsParallel
from .scheduler import LRScheduler
from .tools import TrainerTools

_DEFAULT_MODEL_NAME = "model.pth"
_DEFAULT_STEPS_NAME = 'steps.pt'


def get_rng_state_dict() -> dict:
    state = {
        'torch_rng_state': torch.get_rng_state(),
    }
    device_type = TrainerTools().parallel.device_type
    try:
        if device_type == 'cuda' and torch.cuda.is_available():
            state['device_rng_state'] = torch.cuda.get_rng_state()
        elif device_type == 'npu' and hasattr(torch, 'npu') and torch.npu.is_available():
            state['device_rng_state'] = torch.npu.get_rng_state()
        elif device_type == 'mlu' and hasattr(torch, 'mlu') and torch.mlu.is_available():
            state['device_rng_state'] = torch.mlu.get_rng_state()
        elif device_type == 'mps' and hasattr(torch, 'mps') and torch.backends.mps.is_available():
            state['device_rng_state'] = torch.mps.get_rng_state()
    except Exception: ...
    return state


def restore_rng_state_dict(state: Optional[dict]):
    if not state:
        return
    if 'torch_rng_state' in state and state['torch_rng_state'] is not None:
        torch.set_rng_state(state['torch_rng_state'].cpu().to(torch.uint8))

    if 'device_rng_state' in state and state['device_rng_state'] is not None:
        dev_state = state['device_rng_state'].cpu().to(torch.uint8)
        device_type = TrainerTools().parallel.device_type
        try:
            if device_type == 'cuda' and torch.cuda.is_available():
                torch.cuda.set_rng_state(dev_state)
            elif device_type == 'npu' and hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.set_rng_state(dev_state)
            elif device_type == 'mlu' and hasattr(torch, 'mlu') and torch.mlu.is_available():
                torch.mlu.set_rng_state(dev_state)
            elif device_type == 'mps' and hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.set_rng_state(dev_state)
        except Exception: ...


def save_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        extra_module: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import save_ds_checkpoint
        save_ds_checkpoint(
            model,
            extra_module=extra_module,
            checkpoint_path=checkpoint_path
        )
    else:
        if TrainerTools().parallel.is_main_process:
            ckpt_dir = _get_ckpt_dir(checkpoint_path)
            model_path = os.path.join(ckpt_dir, _DEFAULT_MODEL_NAME)
            ckpt = {
                'model_state_dict': model.state_dict(),
                'rng_state': get_rng_state_dict(),
            }

            if optimizer:
                ckpt.update({'optim_state_dict': optimizer.state_dict()})

            if extra_module:
                ckpt.update({'extra_module_state_dict': extra_module.state_dict()})

            torch.save(ckpt, model_path)


def load_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[Union[torch.device, str]] = None,
        load_module_only: bool = False,
        extra_module: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None
):
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_checkpoint import load_ds_checkpoint
        load_ds_checkpoint(
            model,
            load_module_only=load_module_only,
            extra_module=extra_module,
            checkpoint_path=checkpoint_path
        )
    else:
        ckpt_dir = _get_ckpt_dir(checkpoint_path)
        model_path = os.path.join(ckpt_dir, _DEFAULT_MODEL_NAME)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, weights_only=True, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])

            if not load_module_only:
                if optimizer and 'optim_state_dict' in state_dict:
                    optimizer.load_state_dict(state_dict['optim_state_dict'])

                if extra_module and 'extra_module_state_dict' in state_dict:
                    extra_module.load_state_dict(state_dict['extra_module_state_dict'])

                if 'rng_state' in state_dict:
                    restore_rng_state_dict(state_dict['rng_state'])


def save_steps(
    epoch: int = 0,
    file_idx: int = 0,
    batch_idx: int = 0,
    lr_scheduler: Optional[LRScheduler] = None,
    dataloader_rng_state: Optional[torch.Tensor] = None,
    checkpoint_path: Optional[str] = None
):
    local_rng_state = get_rng_state_dict()
    world_size = TrainerTools().parallel.world_size

    if world_size > 1 and dist.is_initialized():
        all_rng_states = [None for _ in range(world_size)]
        dist.all_gather_object(all_rng_states, local_rng_state)
    else:
        all_rng_states = [local_rng_state]

    if TrainerTools().parallel.is_main_process:
        ckpt_dir = _get_ckpt_dir(checkpoint_path)
        steps_checkpoint_path = os.path.join(ckpt_dir, _DEFAULT_STEPS_NAME)

        ckpt = {
            'epoch': epoch,
            'file_idx': file_idx,
            'batch_idx': batch_idx,
            'rng_state': local_rng_state,
            'all_rng_states': all_rng_states,
        }

        if dataloader_rng_state is not None:
            ckpt['dataloader_rng_state'] = dataloader_rng_state.cpu().to(torch.uint8)

        if lr_scheduler:
            ckpt.update(lr_scheduler.get_ckpt_dict())

        torch.save(ckpt, steps_checkpoint_path)


def load_steps(checkpoint_path: Optional[str] = None) -> Optional[dict]:
    steps_dict = None

    if TrainerTools().parallel.is_main_process:
        ckpt_dir = _get_ckpt_dir(checkpoint_path)
        steps_checkpoint_path = os.path.join(ckpt_dir, _DEFAULT_STEPS_NAME)

        if os.path.exists(steps_checkpoint_path):
            try:
                steps_dict = torch.load(steps_checkpoint_path, weights_only=True)
            except:
                steps_dict = None

    if TrainerTools().parallel.world_size > 1:
        object_list = [steps_dict]
        dist.broadcast_object_list(object_list, src=0)
        steps_dict = object_list[0]
        TrainerTools().parallel.wait('broadcast steps_dict')

    # 每个 rank 提取属于自己的 rng_state，避免非主卡被 Rank 0 的随机状态覆盖
    if steps_dict is not None and 'all_rng_states' in steps_dict:
        rank = max(0, TrainerTools().parallel.global_rank)
        all_rng_states = steps_dict.pop('all_rng_states')
        if isinstance(all_rng_states, list) and rank < len(all_rng_states):
            steps_dict['rng_state'] = all_rng_states[rank]

    return steps_dict


def _get_ckpt_dir(checkpoint_path: Optional[str]):
    if checkpoint_path:
        ckpt_dir = checkpoint_path
    else:
        ckpt_dir = os.environ.get('CHECKPOINT_DIR', './checkpoints')

    if TrainerTools().parallel.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)

    return ckpt_dir
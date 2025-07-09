import os
from typing import Optional, Union
import torch
from torch import nn
from torch.optim import Optimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .tools import TrainerTools

DEFAULT_CHECKPOINT_NAME = "checkpoint.pth"

def save_fsdp_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        suffix: Optional[str] = None
):
    # 未经过测试 参考：https://doc.hfai.high-flyer.cn/haiscale/haiscale_fsdp.html
    # 是否使用rank0_only=True？
    with FSDP.summon_full_params(
            module=model,
            rank0_only=True,
            writeback=False,
            offload_to_cpu=True
    ):
        if TrainerTools().parallel.is_main_process:
            checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
            if suffix:
                checkpoint_name = f"{checkpoint_name}_{suffix}"

            ckpt = {'model_state_dict': model.state_dict()}
            if optimizer:
                ckpt.update({'optim_state_dict': optimizer.state_dict()})

            torch.save(ckpt, checkpoint_name)


def load_fsdp_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        device: Optional[Union[torch.device, str]] = None,
        suffix: Optional[str] = None
):
    checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
    if suffix:
        checkpoint_name = f"{checkpoint_name}_{suffix}"

    with FSDP.summon_full_params(module=model):
        state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        if optimizer:
            optimizer.load_state_dict(state_dict['optim_state_dict'])

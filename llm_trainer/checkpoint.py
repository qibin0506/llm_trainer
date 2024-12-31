import os
from typing import Optional, Union
import torch
from torch import nn
from torch.optim import Optimizer
from .scheduler import LRScheduler
from .train_tools import TrainerTools
from .dcp import save_dcp, load_dcp, convert_dcp_to_pth

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

DEFAULT_CHECKPOINT_NAME = "checkpoint.pth"

def save_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None
):
    if os.environ.get('ENABLE_DCP', '1') == '1':
        save_dcp(model, optimizer)
    else:
        if isinstance(model, FSDP):
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
                    ckpt = {'model_state_dict': model.state_dict()}

                    if optimizer is not None:
                        ckpt.update({'optim_state_dict': optimizer.state_dict()})

                    torch.save(ckpt, checkpoint_name)
        else:
            if TrainerTools().parallel.is_main_process:
                checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
                ckpt = {'model_state_dict': TrainerTools().parallel.raw_model.state_dict()}

                if optimizer is not None:
                    ckpt.update({'optim_state_dict': optimizer.state_dict()})

                torch.save(ckpt, checkpoint_name)

    # 暂时只保存主进程的
    if lr_scheduler is not None and TrainerTools().parallel.is_main_process:
        lr_checkpoint_name = f"{os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)}.lr.steps"
        torch.save({'lr_steps': lr_scheduler.cur_steps}, lr_checkpoint_name)


def load_checkpoint(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
        device: Optional[Union[torch.device, str]] = None
):
    checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)

    if os.environ.get('ENABLE_DCP', '1') == '1':
        if isinstance(model, FSDP):
            load_dcp(model, optimizer)
        else:
            # load_dcp方式在cpu上会报错，所以改为先将ckpt转换为pth，然后再加载pth
            # load_dcp(model, optimizer)
            if os.path.exists(checkpoint_name):
                os.remove(checkpoint_name)

            convert_dcp_to_pth(checkpoint_name)
            ckpt = torch.load(checkpoint_name, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['app']['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(ckpt['app']['optim_state_dict'])
    else:
        if os.path.exists(checkpoint_name):
            # 未经过测试，else的逻辑经过测试在fsdp下也没问题
            if isinstance(model, FSDP):
                with FSDP.summon_full_params(module=model):
                    state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
                    model.load_state_dict(state_dict['model_state_dict'])

                    if optimizer is not None:
                        optimizer.load_state_dict(state_dict['optim_state_dict'])
            else:
                state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])

                if optimizer is not None:
                    optimizer.load_state_dict(state_dict['optim_state_dict'])

    if lr_scheduler is not None:
        lr_checkpoint_name = f"{os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)}.lr.steps"
        if os.path.exists(lr_checkpoint_name):
            lr_checkpoint = torch.load(lr_checkpoint_name, weights_only=True)
            lr_scheduler.update_steps(lr_checkpoint['lr_steps'])

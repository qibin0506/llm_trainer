import os
from typing import Optional, Union, Tuple
import torch
from torch import nn
from torch.optim import Optimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist

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


def _get_fsdp_full_state_dict_on_rank0(model: nn.Module) -> Optional[dict]:
    """
        可以在任意rank上调用，然后只有rank0有值
    """

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    with FSDP.summon_full_params(model, writeback=False, offload_to_cpu=True):
        if TrainerTools().parallel.is_main_process:
            return {k: v.clone() for k, v in model.state_dict().items()}

    return None


def get_fsdp_model_params(model: nn.Module):
    """
        从一个 FSDP 包装的模型中高效地提取完整的 FP32 state_dict。
        这个函数会聚合所有分片的参数，并确保所有 rank 都收到一个完整的副本。
    """

    state_dict = _get_fsdp_full_state_dict_on_rank0(model)

    # 现在，只有 rank 0 上的 state_dict 是一个有效的字典，其他 rank 上是 None。
    # 我们需要将其广播给所有进程。
    if TrainerTools().parallel.world_size > 1:
        # 准备一个列表，rank 0 有数据，其他 rank 是占位符
        object_list = [state_dict] if TrainerTools().parallel.is_main_process else [None]
        # 执行广播，这个操作是阻塞的，会同步所有进程
        dist.broadcast_object_list(object_list, src=0)
        # 所有进程从列表中获取广播后的 state_dict 副本
        state_dict = object_list[0]

    return state_dict

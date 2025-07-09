from typing import Optional
from torch import nn
import torch.distributed as dist

from .tools import TrainerTools

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
except: ...


def _get_ds_full_state_dict_on_rank0(model: DeepSpeedEngine) -> Optional[dict]:
    """
        需要在所有rank上调用，然后只有rank0有值
    """

    if model.zero_optimization_stage() != 3:
        if TrainerTools().parallel.is_main_process:
            return {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
        return None

    # --- ZeRO-3 ---
    # 只调用一次 GatheredParameters，传入所有参数
    with deepspeed.zero.GatheredParameters(model.parameters(), modifier_rank=0):
        if TrainerTools().parallel.is_main_process:
            # 在这个 'with' 代码块内，rank 0 上的 model.module 拥有完整的参数
            # 所以我们可以像操作普通模型一样直接调用 state_dict()
            full_state_dict = model.module.state_dict()

            # 将其克隆到 CPU 并返回
            return {k: v.cpu().clone() for k, v in full_state_dict.items()}

    # 其他 rank 执行到这里时，上下文结束，直接返回 None
    return None

    # # ZeRO-3
    # state_dict_on_rank_0 = {}
    # for param_name, param in model.module.named_parameters():
    #     if hasattr(param, 'ds_id'):
    #         with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
    #             if TrainerTools().parallel.is_main_process:
    #                 state_dict_on_rank_0[param_name] = param.data.to(torch.float32).cpu().clone()
    #     else:
    #         if TrainerTools().parallel.is_main_process:
    #             state_dict_on_rank_0[param_name] = param.data.to(torch.float32).cpu().clone()
    #
    # return state_dict_on_rank_0 if TrainerTools().parallel.is_main_process else None


def get_ds_model_params(model: nn.Module, only_rank0=False):
    """
        从一个正在运行的 DeepSpeedEngine 中高效地提取完整的 FP32 state_dict，
        兼容 ZeRO Stages 0, 1, 2, 3。
        包含了对 ZeRO-3 中分片参数的正确处理。
    """

    assert isinstance(model, DeepSpeedEngine)
    state_dict = _get_ds_full_state_dict_on_rank0(model)

    # 现在，只有 rank 0 上的 state_dict 是一个有效的字典，其他 rank 上是 None。
    # 我们需要将其广播给所有进程。
    if not only_rank0 and TrainerTools().parallel.world_size > 1:
        # 准备一个列表，rank 0 有数据，其他 rank 是占位符
        object_list = [state_dict] if TrainerTools().parallel.is_main_process else [None]
        # 执行广播，这个操作是阻塞的，会同步所有进程
        dist.broadcast_object_list(object_list, src=0)
        # 所有进程从列表中获取广播后的 state_dict 副本
        state_dict = object_list[0]

    return state_dict
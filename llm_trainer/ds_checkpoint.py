import os
from typing import Optional
from glob import glob
import shutil
import torch
from torch import nn
import torch.distributed as dist

from .tools import TrainerTools

try:
    import deepspeed
    from deepspeed import DeepSpeedEngine
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
except: ...

"""
函数	功能	是否加载模型到内存	是否保存到文件 主要用途
get_fp32_state_dict_from_zero_checkpoint	从 ZeRO 检查点提取 FP32 状态字典	否	否	获取模型权重，用于推理、迁移等
load_state_dict_from_zero_checkpoint	从 ZeRO 检查点加载模型和优化器状态	是	否	恢复训练状态，继续训练
convert_zero_checkpoint_to_fp32_state_dict	将 ZeRO 检查点转换为独立的 FP32 状态字典文件	否	是	创建可移植的 FP32 权重文件，用于部署、分享等
"""

def save_ds_checkpoint(
        model: nn.Module,
        suffix: Optional[str] = None
):
    assert isinstance(model, DeepSpeedEngine)
    ckpt_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')
    if suffix:
        ckpt_dir = f"{ckpt_dir}_{suffix}"

    try:
        # 包括model、optimizer等状态
        model.save_checkpoint(save_dir=ckpt_dir)
    except:
        return

    # 删除历史checkpoint
    ckpt_paths = glob(os.path.join(ckpt_dir, "global_*"))
    if len(ckpt_paths) > 2:
        # 按修改时间排序，找到最旧的目录
        oldest_ckpt = sorted(ckpt_paths, key=os.path.getmtime)[0]
        try:
            shutil.rmtree(oldest_ckpt)
        except: ...


def load_ds_checkpoint(
        model: nn.Module,
        load_module_only: bool = False,
        suffix: Optional[str] = None
):
    assert isinstance(model, DeepSpeedEngine)
    ckpt_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')
    if suffix:
        ckpt_dir = f"{ckpt_dir}_{suffix}"

    # 包括model、optimizer等状态
    if os.path.exists(ckpt_dir):
        model.load_checkpoint(load_dir=ckpt_dir, load_module_only=load_module_only)


def load_ds_checkpoint_for_eval(model: nn.Module):
    ckpt_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')
    state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
    model.load_state_dict(state_dict)


def get_ds_model_params(model: nn.Module):
    """
        从一个正在运行的 DeepSpeedEngine 中高效地提取完整的 FP32 state_dict，
        兼容 ZeRO Stages 0, 1, 2, 3。
        这个版本包含了对 ZeRO-3 中非分片参数的正确处理。
    """

    assert isinstance(model, DeepSpeedEngine)
    zero_stage = model.zero_optimization_stage()
    state_dict = None

    if TrainerTools().parallel.is_main_process:
        if zero_stage == 3:
            # ZeRO-3: Rank 0 聚合参数来构建完整的 state_dict
            state_dict = {}
            for param in model.module.parameters():
                # 关键检查：判断参数是否被 ZeRO-3 分片管理
                if hasattr(param, 'ds_id'):
                    # 这是被分片的参数，使用 GatheredParameters 聚合
                    with deepspeed.zero.GatheredParameters(param, modifier_rank=0):
                        # .clone() 创建一个独立副本, .to('cpu') 移动到CPU, .to(torch.float32) 确保类型
                        state_dict[param.ds_name] = param.data.to(torch.float32).cpu().clone()
                else:
                    # 这是未被分片的参数 (e.g., tied weights, buffers), 直接从 Rank 0 复制
                    state_dict[param.ds_name] = param.data.to(torch.float32).cpu().clone()
        else:  # zero_stage in [0, 1, 2]
            # 在这些 stage，rank 0 已经有完整的模型。
            # 我们从 model_engine.module 获取原始模型状态。
            state_dict = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}

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


import os
from glob import glob
import shutil
from torch import nn
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

def save_ds_checkpoint(model: nn.Module):
    assert isinstance(model, DeepSpeedEngine)
    ckpt_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')

    try:
        # 包括model、optimizer等状态
        model.save_checkpoint(save_dir=ckpt_dir)
    except: ...

    # 只在main rank上执行
    if TrainerTools().parallel.is_main_process:
        # 最多保存多少checkpoint，默认为2
        max_to_keep = int(os.environ.get('CKPT_MAX_TO_KEEP', '2'))
        # 删除历史checkpoint
        ckpt_paths = glob(os.path.join(ckpt_dir, "global_*"))
        if len(ckpt_paths) > max_to_keep:
            # 按修改时间排序，找到最旧的目录
            oldest_ckpt = sorted(ckpt_paths, key=os.path.getmtime)[0]
            try:
                shutil.rmtree(oldest_ckpt)
            except: ...

    TrainerTools().parallel.wait('remove old ds checkpoint')


def load_ds_checkpoint(
        model: nn.Module,
        load_module_only: bool = False
):
    assert isinstance(model, DeepSpeedEngine)
    ckpt_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')

    # 包括model、optimizer等状态
    if os.path.exists(ckpt_dir):
        model.load_checkpoint(load_dir=ckpt_dir, load_module_only=load_module_only)


def load_ds_checkpoint_for_eval(model: nn.Module):
    ckpt_dir = os.environ.get('DIST_CHECKPOINT_DIR', 'checkpoint')
    state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_dir)
    model.load_state_dict(state_dict)

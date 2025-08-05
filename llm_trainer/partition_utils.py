from typing import Optional
from contextlib import contextmanager
import itertools
from packaging import version
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from .tools import TrainerTools
from .parallel_ds import DsParallel
from .parallel_ddp import DdpParallel


@contextmanager
def unwrap_model_for_generation(model: nn.Module):
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    Args:
        model:
            Model to be unwrapped.
    Yields:
        Unwrapped model.

    Example:
    ```python
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        generated_outputs = unwrapped_model.generate(input_ids)
    ```
    """
    if isinstance(TrainerTools().parallel, DsParallel):
        import deepspeed
        assert isinstance(model, deepspeed.DeepSpeedEngine)

        if model.zero_optimization_stage() == 3:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                _remove_hooks(model)
                yield unwrap_model(model)
                _add_hooks(model)
        else:
            yield unwrap_model(model)
    elif isinstance(TrainerTools().parallel, DdpParallel):
        yield unwrap_model(model)
    else:
        yield model


def sync_model_params(_from: nn.Module, _to: Optional[nn.Module], mixup_alpha: float = 1.0):
    """
        必须在所有rank上调用，非rank0, _to 可以设置为None.
        当前函数不适用于_to是一个zero3模型
    """
    if isinstance(TrainerTools().parallel, DsParallel):
        state_dict = _get_ds_model_params(_from, only_rank0=_to is None)
    elif isinstance(_from, DDP):
        state_dict = _from.module.state_dict()
    else:
        state_dict = _from.state_dict()

    if not _to or not state_dict:
        return

    unwrap_to_model = unwrap_model(_to)
    if mixup_alpha == 1.0:
        # 直接覆盖
        unwrap_to_model.load_state_dict(state_dict)
    else:
        # 混合参数
        for param_name, target_param in unwrap_to_model.named_parameters():
            if param_name in state_dict:
                from_param_tensor = state_dict[param_name]
                target_param.data.mul_(1.0 - mixup_alpha).add_(
                    from_param_tensor.data.to(target_param.device),
                    alpha=mixup_alpha
                )

    # if isinstance(TrainerTools().parallel, DsParallel):
    #     _sync_ds_model_params(_from, _to, mixup_alpha)
    # elif isinstance(TrainerTools().parallel, DdpParallel):
    #     _sync_ddp_model_params(_from, _to, mixup_alpha)
    # else:
    #     _copy_params(_from, _to, mixup_alpha)


def unwrap_model(model) -> nn.Module:
    try:
        import deepspeed
        if isinstance(model, deepspeed.DeepSpeedEngine):
            return model.module
    except: ...

    if isinstance(model, DDP):
        return model.module

    return model


def _get_ds_full_state_dict_on_rank0(model: nn.Module) -> Optional[dict]:
    """
        需要在所有rank上调用，然后只有rank0有值
    """
    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)

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


def _get_ds_model_params(model: nn.Module, only_rank0=False):
    """
        从一个正在运行的 DeepSpeedEngine 中高效地提取完整的 FP32 state_dict，
        兼容 ZeRO Stages 0, 1, 2, 3。
        包含了对 ZeRO-3 中分片参数的正确处理。
    """

    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)
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


def _copy_params(model, target_model, mixup_alpha):
    for target_param, copy_param in zip(target_model.parameters(), model.parameters()):
        target_param.data.mul_(1.0 - mixup_alpha).add_(copy_param.data, alpha=mixup_alpha)


def _sync_ds_model_params(_from: nn.Module, _to: Optional[nn.Module], mixup_alpha: float = 1.0):
    import deepspeed
    assert isinstance(_from, deepspeed.DeepSpeedEngine)

    origin_from = unwrap_model(_from)

    if _from.zero_optimization_stage() == 3:
        with deepspeed.zero.GatheredParameters(list(origin_from.parameters()) + list(_to.parameters()), modifier_rank=0):
            # why only rank 0?
            if TrainerTools().parallel.is_main_process:
                _copy_params(origin_from, _to, mixup_alpha)
    else:
        _copy_params(origin_from, _to, mixup_alpha)


def _sync_ddp_model_params(_from: nn.Module, _to: Optional[nn.Module], mixup_alpha: float = 1.0):
    assert isinstance(_from, DDP)

    origin_from = unwrap_model(_from)
    _copy_params(origin_from, _to, mixup_alpha)


def _add_hooks(model: nn.Module) -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)

    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):
        # Account for renaming in https://github.com/deepspeedai/DeepSpeed/pull/6847
        optimizer_offload._register_deepspeed_module(optimizer_offload.module)
    else:
        optimizer_offload._register_hooks_recursively(optimizer_offload.module)


def _remove_hooks(model: nn.Module) -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)

    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in _iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def _iter_params(module, recurse=False):
    return [param for _, param in _get_all_parameters(module, recurse)]


def _get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

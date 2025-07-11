from typing import Optional
from contextlib import contextmanager
import itertools
from packaging import version
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

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
    if isinstance(TrainerTools().parallel, DsParallel):
        _sync_ds_model_params(_from, _to, mixup_alpha)
    elif isinstance(TrainerTools().parallel, DdpParallel):
        _sync_ddp_model_params(_from, _to, mixup_alpha)
    else:
        _copy_params(_from, _to, mixup_alpha)


def unwrap_model(model) -> nn.Module:
    try:
        import deepspeed
        if isinstance(model, deepspeed.DeepSpeedEngine):
            return model.module
    except: ...

    if isinstance(model, DDP):
        return model.module

    return model


def _copy_params(model, target_model, mixup_alpha):
    for target_param, copy_param in zip(target_model.parameters(), model.parameters()):
        target_param.data.mul_(1.0 - mixup_alpha).add_(copy_param.data, alpha=mixup_alpha)


def _sync_ds_model_params(_from: nn.Module, _to: Optional[nn.Module], mixup_alpha: float = 1.0):
    import deepspeed
    assert isinstance(_from, deepspeed.DeepSpeedEngine)

    origin_from = unwrap_model(_from)

    if _from.zero_optimization_stage() == 3:
        with deepspeed.zero.GatheredParameters(list(origin_from.parameters()) + list(_to.parameters()), modifier_rank=0):
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

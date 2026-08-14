from typing import Optional
from contextlib import contextmanager
import itertools
from packaging import version
import torch
from torch import nn

from .tools import TrainerTools
from .parallel import DsParallel


@contextmanager
def unwrap_model_for_generation(model: nn.Module):
    """
    解包model用于生成
    """
    if isinstance(TrainerTools().parallel, DsParallel):
        import deepspeed
        assert isinstance(model, deepspeed.DeepSpeedEngine)

        if model.zero_optimization_stage() == 3:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                _remove_hooks(model)
                try:
                    yield unwrap_model(model)
                finally:
                    _add_hooks(model)
        else:
            yield unwrap_model(model)
    else:
        yield model


def sync_model_state_dict(
        _from: nn.Module,
        _to: nn.Module,
        mixup_alpha: float = 1.0,
        max_elements_per_chunk = 100_000_000
):
    """
    同步_from参数到_to，需要在所有rank上调用
    """
    assert _from is not None
    assert _to is not None

    from_model = unwrap_model(_from)
    to_model = unwrap_model(_to)

    is_from_zero3 = False
    is_to_zero3 = False

    if isinstance(TrainerTools().parallel, DsParallel):
        import deepspeed
        if isinstance(_from, deepspeed.DeepSpeedEngine) and _from.zero_optimization_stage() == 3:
            is_from_zero3 = True
        if isinstance(_to, deepspeed.DeepSpeedEngine) and _to.zero_optimization_stage() == 3:
            is_to_zero3 = True

    def _apply_mixup(p_from: torch.Tensor, p_to: torch.Tensor):
        copy_tensor = p_from.to(device=p_to.device, dtype=p_to.dtype)
        if mixup_alpha == 1.0:
            p_to.copy_(copy_tensor)
        else:
            p_to.mul_(1.0 - mixup_alpha).add_(copy_tensor, alpha=mixup_alpha)

    with torch.no_grad():
        from_buffers = dict(from_model.named_buffers())
        for name, buf_to in to_model.named_buffers():
            if name in from_buffers:
                _apply_mixup(from_buffers[name], buf_to)

        from_params = dict(from_model.named_parameters())

        chunk_p_from = []
        chunk_p_to = []
        current_elements = 0

        def _process_sync_chunk(p_from_list, p_to_list):
            if not p_from_list:
                return

            if is_from_zero3 and is_to_zero3:
                import deepspeed
                with deepspeed.zero.GatheredParameters(p_from_list, modifier_rank=None):
                    with deepspeed.zero.GatheredParameters(p_to_list, modifier_rank=0):
                        if TrainerTools().parallel.is_main_process:
                            for pf, pt in zip(p_from_list, p_to_list):
                                _apply_mixup(pf, pt)
            elif is_from_zero3 and not is_to_zero3:
                import deepspeed
                with deepspeed.zero.GatheredParameters(p_from_list, modifier_rank=None):
                    for pf, pt in zip(p_from_list, p_to_list):
                        _apply_mixup(pf, pt)
            elif not is_from_zero3 and is_to_zero3:
                raise NotImplementedError('is_from_zero3=False and is_to_zero3=True is not implemented')
            else:
                for pf, pt in zip(p_from_list, p_to_list):
                    _apply_mixup(pf, pt)

        for name, param_to in to_model.named_parameters():
            if name in from_params:
                param_from = from_params[name]
                chunk_p_from.append(param_from)
                chunk_p_to.append(param_to)

                numel = getattr(param_from, 'ds_numel', param_from.numel())
                current_elements += numel

                if current_elements >= max_elements_per_chunk:
                    _process_sync_chunk(chunk_p_from, chunk_p_to)
                    chunk_p_from = []
                    chunk_p_to = []
                    current_elements = 0

        _process_sync_chunk(chunk_p_from, chunk_p_to)


def get_full_state_dict_on_rank0(
        model: nn.Module,
        max_elements_per_chunk = 100_000_000
):
    """
    在rank0上获取完整参数，需要在所有rank上调用，其他rank返回None
    """
    try:
        import deepspeed
        if isinstance(model, deepspeed.DeepSpeedEngine):
            return _get_ds_full_state_dict_on_rank0(model, max_elements_per_chunk)
    except Exception: ...

    if TrainerTools().parallel.is_main_process:
        return {k: v.cpu().clone() for k, v in unwrap_model(model).state_dict().items()}

    return None


def unwrap_model(model) -> nn.Module:
    try:
        import deepspeed
        if isinstance(model, deepspeed.DeepSpeedEngine):
            return model.module
    except Exception: ...

    return model


# def get_ds_state_dict(
#         model: nn.Module,
#         only_rank0 = False,
#         max_elements_per_chunk = 100_000_000
# ):
#     """
#     从一个正在运行的 DeepSpeedEngine 中高效地提取完整的 FP32 state_dict，兼容 ZeRO Stages 0, 1, 2, 3。
#     需要在所有rank上调用；如果only_rank0为False，则所有rank都会同步获取最新参数，否则，其他rank返回None
#     """
#     import deepspeed
#     assert isinstance(model, deepspeed.DeepSpeedEngine)
#
#     if only_rank0:
#         return _get_ds_full_state_dict_on_rank0(model, max_elements_per_chunk=max_elements_per_chunk)
#
#     if model.zero_optimization_stage() != 3:
#         return {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
#
#     state_dict = {}
#     if TrainerTools().parallel.is_main_process or not only_rank0:
#         for name, buf in model.module.named_buffers():
#             state_dict[name] = buf.cpu().clone()
#
#     chunk_names = []
#     chunk_params = []
#     current_elements = 0
#
#     def _process_chunk(names, params):
#         if not params:
#             return
#         with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
#             for n, p in zip(names, params):
#                 state_dict[n] = p.cpu().clone()
#
#     for name, param in model.module.named_parameters():
#         chunk_names.append(name)
#         chunk_params.append(param)
#
#         numel = getattr(param, 'ds_numel', param.numel())
#         current_elements += numel
#
#         if current_elements >= max_elements_per_chunk:
#             _process_chunk(chunk_names, chunk_params)
#             chunk_names = []
#             chunk_params = []
#             current_elements = 0
#
#     _process_chunk(chunk_names, chunk_params)
#     return state_dict


def _get_ds_full_state_dict_on_rank0(
        model: nn.Module,
        max_elements_per_chunk = 100_000_000
) -> Optional[dict]:
    """
    在rank0上获取完整ds模型参数，需要在所有rank上调用，其他rank返回None
    """
    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)

    if model.zero_optimization_stage() != 3:
        if TrainerTools().parallel.is_main_process:
            return {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
        return None

    state_dict = {}
    state_vars = model.module.state_dict(keep_vars=True)

    if TrainerTools().parallel.is_main_process:
        for name, var in state_vars.items():
            if not isinstance(var, torch.nn.Parameter):
                state_dict[name] = var.cpu().clone()

    unique_params = []
    param_id_to_names = {}

    for name, var in state_vars.items():
        if isinstance(var, torch.nn.Parameter):
            pid = id(var)
            if pid not in param_id_to_names:
                param_id_to_names[pid] = []
                unique_params.append(var)
            param_id_to_names[pid].append(name)

    chunk_params = []
    chunk_names_list = []
    current_elements = 0

    def _process_chunk(params, names_list):
        if not params:
            return
        with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
            if TrainerTools().parallel.is_main_process:
                for p, names in zip(params, names_list):
                    for n in names:
                        state_dict[n] = p.cpu().clone()

    for p in unique_params:
        chunk_params.append(p)
        chunk_names_list.append(param_id_to_names[id(p)])

        numel = getattr(p, 'ds_numel', p.numel())
        current_elements += numel

        if current_elements >= max_elements_per_chunk:
            _process_chunk(chunk_params, chunk_names_list)
            chunk_params = []
            chunk_names_list = []
            current_elements = 0

    _process_chunk(chunk_params, chunk_names_list)
    return state_dict if TrainerTools().parallel.is_main_process else None


def _add_hooks(model: nn.Module) -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    import deepspeed
    assert isinstance(model, deepspeed.DeepSpeedEngine)

    if not hasattr(model, "optimizer") or model.optimizer is None:  # before the first training step, the model has no optimizer
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

    if not hasattr(model, "optimizer") or model.optimizer is None:  # before the first training step, the model has no optimizer
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

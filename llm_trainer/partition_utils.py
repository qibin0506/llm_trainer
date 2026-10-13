from typing import Optional
import itertools
import functools
import types
from contextlib import contextmanager, nullcontext
from packaging import version

import torch
from torch import nn

from .tools import TrainerTools
from .parallel import DsParallel


@contextmanager
def unwrap_model_for_generation(model: nn.Module):
    unwrapped_model = unwrap_model(model)
    is_gradient_checkpointing = unwrapped_model.is_gradient_checkpointing

    if is_gradient_checkpointing:
        unwrapped_model.gradient_checkpointing_disable()

    try:
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
                yield unwrapped_model
        else:
            yield unwrapped_model
    finally:
        if is_gradient_checkpointing:
            unwrapped_model.gradient_checkpointing_enable()


@contextmanager
def zero3_trace_guard(model: nn.Module):
    """
    DeepSpeed ZeRO-3 状态机与 Trace 保护上下文：
    保持参数逐层分片的前提下，在退出时清理状态机并作废残缺 Trace，
    杜绝无 Backward 的前向污染随后的训练反向传播流水线。
    非 ZeRO-3 环境下直接透传，零开销。
    """
    unwrapped = unwrap_model(model)
    was_gradient_checkpointing = unwrapped.is_gradient_checkpointing

    if was_gradient_checkpointing:
        unwrapped.gradient_checkpointing_disable()

    try:
        yield
    finally:
        _reset_zero3_state_machine(model)
        if was_gradient_checkpointing:
            unwrapped.gradient_checkpointing_enable()


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
    except (ImportError, ModuleNotFoundError):
        pass

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


def maybe_gather_lm_head_ctx(*params: Optional[torch.nn.Parameter]):
    """
    上下文管理器：当且仅当 DeepSpeed ZeRO-3 启用时，自动 AllGather 分片的 lm_head 权重/偏置。
    非 ZeRO-3 环境下直接返回 nullcontext()，零额外开销。
    """
    is_zero3 = False
    try:
        import deepspeed
        if any(p is not None and hasattr(p, "ds_id") and deepspeed.zero.partition_parameters.is_zero_param(p) for p in params):
            is_zero3 = True
    except Exception:
        pass

    if not is_zero3:
        return nullcontext()

    try:
        import deepspeed
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

        to_gather = {
            id(p): p for p in params
            if p is not None and getattr(p, 'ds_status', None) != ZeroParamStatus.AVAILABLE
        }

        if not to_gather:
            return nullcontext()

        return deepspeed.zero.GatheredParameters(list(to_gather.values()))
    except:
        return nullcontext()


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

    if hasattr(optimizer_offload, "param_coordinator"):
        coordinator = optimizer_offload.param_coordinator
        if not coordinator.is_invalid_trace():
            coordinator._invalidate_trace()

    if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):
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
    ext_params = sub_module.ds_external_parameters() if hasattr(sub_module, "ds_external_parameters") else []
    return itertools.chain(sub_module.named_parameters(recurse=recurse), ext_params)


def _reset_zero3_state_machine(model: nn.Module) -> None:
    """
    清洗并复位 DeepSpeed ZeRO-3 训练引擎的参数状态机与 Trace 队列：
    1. 确保所有正向展开过的参数完全释放并重新分片 (回到 NOT_AVAILABLE 状态)；
    2. 清理各参数残留的 ds_active_sub_modules 激活标记；
    3. 强制作废无 Backward 期间录制的残缺 Trace，确保后续带梯度的训练 Forward 从头正确录制。
    """
    if not isinstance(TrainerTools().parallel, DsParallel):
        return
    try:
        import deepspeed
        if not isinstance(model, deepspeed.DeepSpeedEngine) or model.zero_optimization_stage() != 3:
            return

        optimizer = getattr(model, "optimizer", None)
        if optimizer is None:
            return

        optimizer_offload = getattr(optimizer, "parameter_offload", optimizer)
        # 1. 确保所有正向分片参数全部归位为 NOT_AVAILABLE
        if hasattr(optimizer_offload, "partition_all_parameters"):
            optimizer_offload.partition_all_parameters()

        # 2. 清理所有子模块残留的激活标记，防止生命周期计数泄漏
        for param in _iter_params(optimizer_offload.module, recurse=True):
            if hasattr(param, "ds_active_sub_modules"):
                param.ds_active_sub_modules.clear()

        # 3. 核心：强制作废残缺 Trace 队列
        if hasattr(optimizer_offload, "param_coordinator"):
            coordinator = optimizer_offload.param_coordinator
            if hasattr(coordinator, "is_invalid_trace") and not coordinator.is_invalid_trace():
                coordinator._invalidate_trace()
    except Exception:
        pass


# 必选在maybe_gather_lm_head_ctx后面import
from .loss import chunked_log_probs, ChunkedLMLoss


def add_fused_lm_head(model: nn.Module, chunk_size: int):
    """
    在模型被 DeepSpeed Engine 包装之前对 model.forward 进行 Monkey-Patch，
    实现主干前向、分块 Loss 计算以及分块 LogProbs 计算的闭环封装。

    1. 当 fused_lm_head=True 且 labels is not None 时 (SFT / Pretrain 模式)：
      - 强制主干执行 return_logits=False (节约大量中间显存)；
      - 在 model.forward 内部直接访问 self.lm_head 并通过 ChunkedLMLoss / ChunkedCrossEntropyFunction 计算损失；
      - 返回包含 'loss' 和 'ce_loss' 的字典，使 DeepSpeed Engine 的输出 backward hooks
        直接挂在 'loss' 根节点上，彻底解决孤儿参数和梯度累积尺度撕裂问题。
    2. 当 fused_lm_head=True 且 completion_ids is not None 时 (PPO / GRPO 模式)：
      - 强制主干执行 return_logits=False (彻底规避 [B, S, V] 巨额 Logits 物化)；
      - 在 model.forward 内部直接访问 self.lm_head 并通过 ChunkedLogProbFunction 计算 Completion LogProbs；
      - 返回包含 'log_probs' 的字典，Autograd 计算图直接闭环绑定到 lm_head.weight 与主干；
    3. 当 fused_lm_head=False 时：
      - 完全保留原模型 forward 行为（推理/评测/生成无缝兼容）。
    """
    unwrapped = unwrap_model(model)
    if getattr(unwrapped, "_has_fused_lm_head", False):
        return unwrapped

    original_forward = unwrapped.forward

    @functools.wraps(original_forward)
    def _fused_forward(
        self,
        *args,
        fused_lm_head: bool = False,
        labels: Optional[torch.Tensor] = None,
        completion_ids: Optional[torch.Tensor] = None,
        completion_mask: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        kd_coef: float = 0.0,
        **kwargs
    ):
        if not fused_lm_head or (labels is None and completion_ids is None):
            if labels is not None:
                kwargs["labels"] = labels
            if completion_ids is not None:
                kwargs["completion_ids"] = completion_ids
            if completion_mask is not None:
                kwargs["completion_mask"] = completion_mask
            return original_forward(*args, **kwargs)

        lm_head = getattr(self, "lm_head", None)
        assert lm_head is not None, "Model must have lm_head for fused computation"

        # 闭环分块计算：强制主干不生成全局大 logits，节约巨额显存
        kwargs["return_logits"] = False
        outputs = original_forward(*args, **kwargs)
        hidden_states = outputs["hidden_states"]

        fused_outputs = dict(outputs)

        # 模式 1: Fused Completion LogProbs (PPO / GRPO 模式)
        if completion_ids is not None:
            comp_len = completion_ids.size(1)
            # 因果自回归：预测 completion_ids 的 hidden_states 是其前驱位置
            hidden_comp = hidden_states[:, :-1, :][:, -comp_len:, :]

            # Active Token Filtering: 过滤 Padding Token 冗余投影开销
            targets = completion_ids.clone()
            att_mask = kwargs.get("attention_mask", None)
            if att_mask is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
                att_mask = args[1]

            if completion_mask is not None:
                targets = targets.masked_fill(~completion_mask.bool(), -100)
            elif att_mask is not None and att_mask.dim() == 2 and att_mask.size(1) >= comp_len:
                comp_att_mask = att_mask[:, -comp_len:].bool()
                targets = targets.masked_fill(~comp_att_mask, -100)
            else:
                targets = targets.masked_fill(targets == TrainerTools().tokenizer.pad, -100)

            log_probs = chunked_log_probs(
                hidden_states=hidden_comp,
                lm_head_weight=lm_head.weight,
                targets=targets,
                lm_head_bias=getattr(lm_head, "bias", None),
                chunk_size=chunk_size,
                ignore_index=-100
            )
            for k, v in list(fused_outputs.items()):
                if isinstance(v, torch.Tensor) and k not in ("aux_loss",):
                    fused_outputs[k] = v.detach()
            fused_outputs["log_probs"] = log_probs
            return fused_outputs

        # 模式 2: Fused CE / KD Loss (SFT / Pretrain 模式)
        criterion = getattr(self, "_chunked_lm_loss", None)
        if criterion is None:
            criterion = ChunkedLMLoss(chunk_size=chunk_size)
            self._chunked_lm_loss = criterion

        total_loss, ce_loss = criterion(
            hidden_states=hidden_states,
            lm_head_weight=lm_head.weight,
            labels=labels,
            lm_head_bias=getattr(lm_head, "bias", None),
            teacher_logits=teacher_logits,
            kd_coef=kd_coef,
        )

        for k, v in list(fused_outputs.items()):
            if isinstance(v, torch.Tensor) and k not in ("aux_loss",):
                fused_outputs[k] = v.detach()
        fused_outputs["loss"] = total_loss
        fused_outputs["ce_loss"] = ce_loss.detach() if isinstance(ce_loss, torch.Tensor) else ce_loss
        return fused_outputs

    unwrapped.forward = types.MethodType(_fused_forward, unwrapped)
    unwrapped._has_fused_lm_head = True
    return unwrapped


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

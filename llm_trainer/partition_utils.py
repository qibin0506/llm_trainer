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
        if mixup_alpha == 1.0 or not p_to.is_floating_point():
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
                import deepspeed
                with deepspeed.zero.GatheredParameters(p_to_list, modifier_rank=0):
                    if TrainerTools().parallel.is_main_process:
                        for pf, pt in zip(p_from_list, p_to_list):
                            _apply_mixup(pf, pt)
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
    except (ImportError, ModuleNotFoundError): ...

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
    上下文管理器：当且仅当 DeepSpeed ZeRO-3 启用且参数被分片时，自动 AllGather 分片的 lm_head 权重/偏置。
    - 按 id(p) 去重，避免共享权重重复 gather；
    - 已处于 AVAILABLE 状态的参数（如 tied embeddings 已被 embed_tokens 展开）不重复 gather，避免退出时误触发 re-partition。
    """
    try:
        import deepspeed
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except ImportError:
        return nullcontext()

    to_gather = {
        id(p): p for p in params
        if p is not None and hasattr(p, "ds_id") and getattr(p, "ds_status", None) != ZeroParamStatus.AVAILABLE
    }

    if not to_gather:
        return nullcontext()

    return deepspeed.zero.GatheredParameters(list(to_gather.values()))


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
            if not (hasattr(var, 'ds_id') or isinstance(var, torch.nn.Parameter)):
                state_dict[name] = var.cpu().clone()

    unique_params = []
    param_id_to_names = {}

    for name, var in state_vars.items():
        if hasattr(var, 'ds_id') or isinstance(var, torch.nn.Parameter):
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
    optimizer_offload = _get_zero3_param_offload(model)
    if optimizer_offload is None:
        return

    import deepspeed
    if hasattr(optimizer_offload, "param_coordinator"):
        coordinator = optimizer_offload.param_coordinator
        if hasattr(coordinator, "is_invalid_trace") and not coordinator.is_invalid_trace():
            coordinator._invalidate_trace()

    if hasattr(optimizer_offload, "_register_deepspeed_module"):
        optimizer_offload._register_deepspeed_module(optimizer_offload.module)
    elif hasattr(optimizer_offload, "_register_hooks_recursively"):
        optimizer_offload._register_hooks_recursively(optimizer_offload.module)


def _remove_hooks(model: nn.Module) -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    optimizer_offload = _get_zero3_param_offload(model)
    if optimizer_offload is None:
        return

    for param in _iter_params(optimizer_offload.module, recurse=True):
        if hasattr(param, "ds_active_sub_modules"):
            param.ds_active_sub_modules.clear()

    for hook in getattr(optimizer_offload, "forward_hooks", []):
        hook.remove()
    for hook in getattr(optimizer_offload, "backward_hooks", []):
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def _get_zero3_param_offload(model: nn.Module):
    """
    获取 DeepSpeed ZeRO-3 的参数协调器宿主对象 (兼容含 optimizer 的训练引擎与 optimizer=None 的 ref_model 引擎)。
    """
    import deepspeed
    if not isinstance(model, deepspeed.DeepSpeedEngine) or model.zero_optimization_stage() != 3:
        return None

    optimizer = getattr(model, "optimizer", None)
    if optimizer is not None:
        if hasattr(optimizer, "parameter_offload"):
            return optimizer.parameter_offload
        return optimizer

    if hasattr(model, "parameter_offload") and model.parameter_offload is not None:
        return model.parameter_offload

    return None


def _iter_params(module, recurse=False):
    return [param for _, param in _get_all_parameters(module, recurse)]


def _get_all_parameters(sub_module, recurse=False):
    ext_params = sub_module.ds_external_parameters() if hasattr(sub_module, "ds_external_parameters") else []
    return itertools.chain(sub_module.named_parameters(recurse=recurse), ext_params)


def _reset_zero3_state_machine(model: nn.Module) -> None:
    """
    清洗并复位 DeepSpeed ZeRO-3 引擎（含 train_model 与 optimizer=None 的 ref_model）在 no_grad 前向后的参数状态机与 Trace：
    1. 清理各参数残留的 ds_active_sub_modules 激活标记；
    2. 确保非持久化分片参数安全释放归位为 NOT_AVAILABLE；
    3. 强制作废残缺 Trace 队列 (进入 INVALID 状态，等待下一次 forward 自动重建)；
    """
    if not isinstance(TrainerTools().parallel, DsParallel):
        return
    try:
        optimizer_offload = _get_zero3_param_offload(model)
        if optimizer_offload is None:
            return

        # 1. 清理所有子模块残留的激活标记，防止生命周期计数泄漏阻碍参数释放
        for param in _iter_params(optimizer_offload.module, recurse=True):
            if hasattr(param, "ds_active_sub_modules"):
                param.ds_active_sub_modules.clear()

        # 2. 释放非持久化的正向展开参数
        if hasattr(optimizer_offload, "partition_all_parameters"):
            optimizer_offload.partition_all_parameters()

        # 3. 强制作废无 backward 的残缺 Trace 队列
        if hasattr(optimizer_offload, "param_coordinator"):
            coordinator = optimizer_offload.param_coordinator
            if hasattr(coordinator, "is_invalid_trace") and not coordinator.is_invalid_trace():
                coordinator._invalidate_trace()
    except Exception as e:
        from .log import Logger
        Logger.std_log(f"WARN: _reset_zero3_state_machine encountered error: {e}")

ChunkedLogProbFunction
# 必选在maybe_gather_lm_head_ctx后面import
from .loss import ChunkedLogProbFunction, ChunkedLMLoss


def add_fused_loss_head(model: nn.Module, chunk_size: int):
    """
    在模型被 DeepSpeed Engine 包装之前对 model.forward 进行 Monkey-Patch，
    实现主干前向与分块 Loss (ChunkedCrossEntropy / KD) 计算的闭环封装。
    """
    unwrapped = unwrap_model(model)
    unwrapped._fused_loss_chunk_size = chunk_size
    if getattr(unwrapped, "_has_fused_loss_head", False):
        return unwrapped

    original_forward = unwrapped.forward
    chunked_loss_fn = ChunkedLMLoss(chunk_size=chunk_size)

    @functools.wraps(original_forward)
    def _fused_loss_forward(
        self,
        *args,
        fused_loss: bool = False,
        fused_lm_head: bool = False,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        kd_coef: float = 0.0,
        return_logits: bool = True,
        output_hidden_states: bool = False,
        **kwargs
    ):
        use_fused = fused_loss or fused_lm_head
        curr_chunk_size = getattr(self, "_fused_loss_chunk_size", chunk_size)
        if not use_fused or labels is None:
            if labels is not None:
                kwargs["labels"] = labels
            return original_forward(*args, return_logits=return_logits, **kwargs)

        lm_head = getattr(self, "lm_head", None)
        assert lm_head is not None, "Model must have lm_head for fused loss computation"

        outputs = original_forward(*args, return_logits=False, **kwargs)
        hidden_states = outputs["hidden_states"]

        chunked_loss_fn.chunk_size = curr_chunk_size
        total_loss, ce_loss = chunked_loss_fn(
            hidden_states=hidden_states,
            lm_head_weight=lm_head.weight,
            labels=labels,
            lm_head_bias=getattr(lm_head, "bias", None),
            teacher_logits=teacher_logits,
            kd_coef=kd_coef,
        )

        fused_outputs = dict(outputs)
        if not output_hidden_states:
            fused_outputs["hidden_states"] = None
        fused_outputs["loss"] = total_loss
        fused_outputs["ce_loss"] = ce_loss.detach() if isinstance(ce_loss, torch.Tensor) else ce_loss
        return fused_outputs

    unwrapped.forward = types.MethodType(_fused_loss_forward, unwrapped)
    unwrapped._has_fused_loss_head = True
    return unwrapped


def add_fused_log_prob_head(model: nn.Module, chunk_size: int):
    """
    在模型被 DeepSpeed Engine 包装之前对 model.forward 进行 Monkey-Patch，
    实现主干前向与分块 LogProbs (ChunkedLogProbFunction) 计算的闭环封装。
    - 无条件执行 masked_scatter，保证全 -100 微批次在反向传播中保持计算图连通，彻底消除 ZeRO-3 死锁；
    - 优先使用 completion_mask 决定 completion 区域的 labels，防止 pad_token_id == eos_token_id 时首个 EOS 被 att_mask 误置为 -100。
    """
    unwrapped = unwrap_model(model)
    unwrapped._fused_log_prob_chunk_size = chunk_size
    if getattr(unwrapped, "_has_fused_log_prob_head", False):
        return unwrapped

    original_forward = unwrapped.forward

    @functools.wraps(original_forward)
    def _fused_log_prob_forward(
        self,
        *args,
        fused_log_probs: bool = False,
        fused_lm_head: bool = False,
        labels: Optional[torch.Tensor] = None,
        completion_ids: Optional[torch.Tensor] = None,
        completion_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        return_logits: bool = True,
        output_hidden_states: bool = False,
        **kwargs
    ):
        use_fused = fused_log_probs or fused_lm_head
        curr_chunk_size = getattr(self, "_fused_log_prob_chunk_size", chunk_size)
        if not use_fused or (labels is None and completion_ids is None):
            if labels is not None:
                kwargs["labels"] = labels
            if completion_ids is not None:
                kwargs["completion_ids"] = completion_ids
            if completion_mask is not None:
                kwargs["completion_mask"] = completion_mask
            return original_forward(*args, return_logits=return_logits, **kwargs)

        lm_head = getattr(self, "lm_head", None)
        assert lm_head is not None, "Model must have lm_head for fused log_probs computation"

        outputs = original_forward(*args, return_logits=False, **kwargs)
        hidden_states = outputs["hidden_states"]

        comp_len = completion_ids.size(1) if completion_ids is not None else None
        if labels is None and completion_ids is not None:
            inp_ids = kwargs.get("input_ids", None)
            if inp_ids is None and len(args) > 0 and isinstance(args[0], torch.Tensor):
                inp_ids = args[0]
            if inp_ids is not None:
                labels = inp_ids.clone()
                labels[:, :-comp_len] = -100
                if completion_mask is not None:
                    # 当显式提供 completion_mask (如 loss_mask) 时，直接由 completion_mask 决定有效位置，
                    # 避免 pad_token_id == eos_token_id 时首个 EOS token 被 att_mask == 0 提前抹为 -100
                    labels[:, -comp_len:] = labels[:, -comp_len:].masked_fill(~completion_mask.bool(), -100)
                else:
                    att_mask = kwargs.get("attention_mask", None)
                    if att_mask is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
                        att_mask = args[1]
                    if att_mask is not None:
                        labels[:, -comp_len:] = labels[:, -comp_len:].masked_fill(att_mask[:, -comp_len:] == 0, -100)
            else:
                raise ValueError("input_ids must be provided to compute completion log_probs")
        elif labels is not None and completion_mask is not None and comp_len is not None:
            labels = labels.clone()
            labels[:, -comp_len:] = labels[:, -comp_len:].masked_fill(~completion_mask.bool(), -100)

        assert labels is not None, "labels must be provided or derivable in fused log_probs mode"

        # 直接切片后用 mask 索引（布尔索引自动返回连续张量，避免全量 [B, S-1, H] .contiguous() 显存拷贝）
        hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]

        weight = lm_head.weight
        bias = getattr(lm_head, "bias", None)
        vocab_size = weight.ds_shape[0] if hasattr(weight, "ds_shape") else weight.size(0)
        mask = (shift_labels != -100) & (shift_labels >= 0) & (shift_labels < vocab_size)

        with maybe_gather_lm_head_ctx(weight, bias):
            per_token_logps = ChunkedLogProbFunction.apply(
                hidden[mask],
                weight,
                bias,
                shift_labels[mask],
                curr_chunk_size,
                -100,
                temperature
            )

        # 即使当前 rank 的 mask 全为 False (per_token_logps 形状为 [0])，也必须保持 full_log_probs 连接到
        # ChunkedLogProbFunctionBackward，确保 backward 时所有 rank 同步进入 maybe_gather_lm_head_ctx，杜绝 NCCL 死锁
        full_log_probs = per_token_logps.new_zeros(shift_labels.shape).masked_scatter(mask, per_token_logps)

        fused_outputs = dict(outputs)
        if not output_hidden_states:
            fused_outputs["hidden_states"] = None
        if comp_len is not None:
            fused_outputs["log_probs"] = full_log_probs[:, -comp_len:]
        else:
            fused_outputs["log_probs"] = full_log_probs

        return fused_outputs

    unwrapped.forward = types.MethodType(_fused_log_prob_forward, unwrapped)
    unwrapped._has_fused_log_prob_head = True
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

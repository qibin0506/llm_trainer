from typing import Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
from .partition_utils import maybe_gather_lm_head_ctx
from .utils import log_softmax


def _get_compute_dtype(hidden: torch.Tensor) -> torch.dtype:
    device_type = hidden.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return hidden.dtype


def _addmm_fp32(acc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> None:
    """
    fp32 累加 `acc += a @ b`，防止分块半精度累加导致 entropy drift。
    当 a, b 同为 bf16/fp16 时直接调用 Tensor Core FP32 累加。
    """
    if acc.device.type == "cuda" and a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype:
        try:
            torch.addmm(acc, a, b, out_dtype=torch.float32, out=acc)
            return
        except: ...
    acc.addmm_(a.float(), b.float())


class ChunkedCrossEntropyFunction(torch.autograd.Function):
    """
    1. 前向：由外部调用模块（如 ChunkedLMLoss）统一包裹 maybe_gather_lm_head_ctx 保护 .apply() 调用，
       算子内部直接执行分块 GEMM，累加标量损失，不保存任何中间 Logits，规避多层嵌套破坏 ZeRO-3 状态机。
    2. 反向：在 backward 内部统一进入一次 maybe_gather_lm_head_ctx，保证分块重算与 GEMM 累积期间权重处于 AVAILABLE 状态且全卡对齐。
    3. 显式记录 compute_dtype 并全程采用 FP32 累加 grad_hidden 与 grad_weight
    """
    @staticmethod
    def forward(
            ctx,
            hidden: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            labels: torch.Tensor,
            teacher_logits: Optional[torch.Tensor],
            chunk_size: int,
            ignore_index: int,
            kd_coef: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.kd_coef = kd_coef
        compute_dtype = _get_compute_dtype(hidden)
        ctx.compute_dtype = compute_dtype
        has_teacher = (teacher_logits is not None) and (kd_coef > 0.0)

        orig_h_shape = hidden.shape
        if hidden.dim() == 3:
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            labels_flat = labels.reshape(-1)
            t_logits_flat = teacher_logits.reshape(-1, teacher_logits.shape[-1]) if teacher_logits is not None else None
        else:
            hidden_flat = hidden
            labels_flat = labels
            t_logits_flat = teacher_logits

        vocab_size = weight.ds_shape[0] if hasattr(weight, "ds_shape") else weight.shape[0]
        valid_mask = (labels_flat != ignore_index) & (labels_flat >= 0) & (labels_flat < vocab_size)
        is_all_valid = bool(valid_mask.all().item()) if valid_mask.numel() > 0 else True

        if is_all_valid:
            valid_indices = None
            n_valid = hidden_flat.size(0)
            hidden_valid = hidden_flat
            labels_valid = labels_flat
        else:
            valid_indices = torch.nonzero(valid_mask).squeeze(-1)
            n_valid = valid_indices.numel()
            hidden_valid = hidden_flat[valid_indices]
            labels_valid = labels_flat[valid_indices]

        total_loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
        ce_loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)

        if n_valid > 0:
            w_compute = weight.to(compute_dtype)
            b_float = bias.float() if bias is not None else None

            for start in range(0, n_valid, chunk_size):
                end = min(start + chunk_size, n_valid)
                h_c = hidden_valid[start:end].to(compute_dtype)
                lbl_c = labels_valid[start:end]

                logits_c = (h_c @ w_compute.t()).float()
                if b_float is not None:
                    logits_c = logits_c + b_float

                log_p_c = F.log_softmax(logits_c, dim=-1)
                chunk_ce = F.nll_loss(log_p_c, lbl_c, ignore_index=ignore_index, reduction="sum")
                ce_loss_sum = ce_loss_sum + chunk_ce

                if has_teacher:
                    t_slice = t_logits_flat[start:end] if is_all_valid else t_logits_flat[valid_indices[start:end]]
                    t_logits_c = t_slice.to(device=hidden.device, dtype=torch.float32)
                    t_probs_c = F.softmax(t_logits_c, dim=-1)
                    safe_log_p = torch.clamp(log_p_c, min=-100.0)
                    prod = torch.where(t_probs_c > 0, t_probs_c * safe_log_p, torch.zeros_like(safe_log_p))
                    chunk_kd = -prod.sum()
                    chunk_total = (1.0 - kd_coef) * chunk_ce + kd_coef * chunk_kd
                else:
                    chunk_total = chunk_ce

                total_loss_sum = total_loss_sum + chunk_total

        t_logits_cpu = None
        offload_event = None
        if has_teacher and n_valid > 0:
            t_logits_valid = (t_logits_flat if is_all_valid else t_logits_flat[valid_indices]).contiguous()
            if t_logits_valid.is_cuda:
                pin_success = False
                if torch.cuda.is_available():
                    try:
                        t_logits_cpu = torch.empty(
                            t_logits_valid.shape,
                            dtype=t_logits_valid.dtype,
                            device="cpu",
                            pin_memory=True,
                        )
                        pin_success = True
                    except Exception:
                        t_logits_cpu = torch.empty(
                            t_logits_valid.shape,
                            dtype=t_logits_valid.dtype,
                            device="cpu",
                            pin_memory=False,
                        )
                else:
                    t_logits_cpu = torch.empty(
                        t_logits_valid.shape,
                        dtype=t_logits_valid.dtype,
                        device="cpu",
                    )
                t_logits_cpu.copy_(t_logits_valid, non_blocking=pin_success)
                if pin_success:
                    current_stream = torch.cuda.current_stream(t_logits_valid.device)
                    t_logits_valid.record_stream(current_stream)
                    offload_event = torch.cuda.Event()
                    offload_event.record(current_stream)
                del t_logits_valid
            else:
                t_logits_cpu = t_logits_valid

        denom = max(n_valid, 1)
        total_loss = total_loss_sum / denom
        ce_loss = (ce_loss_sum / denom).detach()

        del t_logits_flat

        ctx.is_all_valid = is_all_valid
        ctx.save_for_backward(hidden_valid, weight, bias, labels_valid, valid_indices)

        ctx.teacher_logits_cpu = t_logits_cpu
        ctx.offload_event = offload_event
        ctx.n_valid = n_valid
        ctx.orig_h_shape = orig_h_shape
        ctx.hidden_dtype = hidden.dtype
        ctx.mark_non_differentiable(ce_loss)
        return total_loss, ce_loss

    @staticmethod
    def backward(ctx, grad_total_loss: Optional[torch.Tensor], grad_ce_loss: Optional[torch.Tensor] = None):
        hidden_valid, weight, bias, labels_valid, valid_indices = ctx.saved_tensors
        t_logits_cpu = getattr(ctx, "teacher_logits_cpu", None)
        offload_event = getattr(ctx, "offload_event", None)
        n_valid = ctx.n_valid
        orig_h_shape = getattr(ctx, "orig_h_shape", hidden_valid.shape)
        hidden_dtype = getattr(ctx, "hidden_dtype", hidden_valid.dtype)
        compute_dtype = getattr(ctx, "compute_dtype", hidden_dtype)
        is_all_valid = getattr(ctx, "is_all_valid", False)

        needs_h_grad = ctx.needs_input_grad[0]
        needs_w_grad = ctx.needs_input_grad[1]
        needs_b_grad = bias is not None and ctx.needs_input_grad[2]

        chunk_size = ctx.chunk_size
        kd_coef = ctx.kd_coef
        has_teacher = (t_logits_cpu is not None) and (kd_coef > 0.0)

        with maybe_gather_lm_head_ctx(weight, bias):
            grad_hidden = torch.zeros(orig_h_shape, dtype=torch.float32, device=hidden_valid.device) if needs_h_grad else None
            grad_hidden_flat = grad_hidden.view(-1, orig_h_shape[-1]) if grad_hidden is not None else None
            grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if needs_w_grad else None
            grad_bias = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device) if needs_b_grad else None

            if grad_total_loss is not None and n_valid > 0:
                g_out = grad_total_loss.float()
                scale = g_out / max(n_valid, 1)

                if has_teacher:
                    if offload_event is not None:
                        if hidden_valid.is_cuda:
                            torch.cuda.current_stream(hidden_valid.device).wait_event(offload_event)
                        offload_event.synchronize()
                    elif hidden_valid.is_cuda and torch.cuda.is_available():
                        torch.cuda.current_stream(hidden_valid.device).synchronize()

                w_compute = weight.to(compute_dtype)
                b_float = bias.float() if bias is not None else None

                for start in range(0, n_valid, chunk_size):
                    end = min(start + chunk_size, n_valid)
                    h_c = hidden_valid[start:end].to(compute_dtype)
                    lbl_c = labels_valid[start:end]

                    logits_c = (h_c @ w_compute.t()).float()
                    if b_float is not None:
                        logits_c = logits_c + b_float

                    d_logits = F.softmax(logits_c, dim=-1)

                    if has_teacher:
                        t_logits_c = t_logits_cpu[start:end].to(hidden_valid.device, non_blocking=False).float()
                        t_probs_c = F.softmax(t_logits_c, dim=-1)
                        d_logits.sub_(t_probs_c.mul_(kd_coef))
                        d_logits.scatter_add_(
                            dim=1,
                            index=lbl_c.unsqueeze(1),
                            src=torch.full((lbl_c.shape[0], 1), -(1.0 - kd_coef), dtype=d_logits.dtype, device=d_logits.device),
                        )
                    else:
                        d_logits.scatter_add_(
                            dim=1,
                            index=lbl_c.unsqueeze(1),
                            src=torch.full((lbl_c.shape[0], 1), -1.0, dtype=d_logits.dtype, device=d_logits.device),
                        )

                    d_logits.mul_(scale)
                    if needs_b_grad:
                        grad_bias.add_(d_logits.sum(dim=0))

                    d_logits_c = d_logits.to(compute_dtype)
                    if needs_h_grad:
                        if is_all_valid:
                            # 连续切片返回 View，可直接原地 FP32 累加
                            _addmm_fp32(grad_hidden_flat[start:end], d_logits_c, w_compute)
                        else:
                            idx_c = valid_indices[start:end]
                            grad_h_c = torch.zeros((end - start, orig_h_shape[-1]), dtype=torch.float32, device=hidden_valid.device)
                            _addmm_fp32(grad_h_c, d_logits_c, w_compute)
                            grad_hidden_flat[idx_c] = grad_h_c
                    if needs_w_grad:
                        _addmm_fp32(grad_weight, d_logits_c.t(), h_c)

        if hasattr(ctx, "teacher_logits_cpu"):
            del ctx.teacher_logits_cpu
        if hasattr(ctx, "offload_event"):
            del ctx.offload_event

        return (
            grad_hidden.to(hidden_dtype) if grad_hidden is not None else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None, None, None, None, None
        )


class ChunkedLogProbFunction(torch.autograd.Function):
    """
    分块计算目标 token 的 log_probs，彻底避免物化 [B, S, V] 全量 Logits。
    - 统一捕获 compute_dtype 并在前反向中对齐矩阵乘精度；
    - 反向传播中 grad_hidden 与 grad_weight 均采用 FP32 缓冲区配合 _addmm_fp32 累加；
    - 区分连续切片视图 (is_all_valid=True) 与花式索引拷贝 (is_all_valid=False)，杜绝花式索引导致梯度丢失。
    """
    @staticmethod
    def forward(
            ctx,
            hidden: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            targets: torch.Tensor,
            chunk_size: int = 256,
            ignore_index: int = -100,
            temperature: float = 1.0,
            vocab_chunk_size: Optional[int] = 32768,
    ) -> torch.Tensor:
        ctx.set_materialize_grads(False)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.temperature = max(temperature, 1e-6)
        compute_dtype = _get_compute_dtype(hidden)
        ctx.compute_dtype = compute_dtype

        orig_targets_shape = targets.shape
        orig_h_shape = hidden.shape
        if hidden.dim() == 3:
            hidden_flat = hidden.reshape(-1, hidden.shape[-1])
            targets_flat = targets.reshape(-1)
        else:
            hidden_flat = hidden
            targets_flat = targets

        log_probs_flat = torch.zeros(targets_flat.shape, dtype=torch.float32, device=hidden.device)

        vocab_size = weight.ds_shape[0] if hasattr(weight, "ds_shape") else weight.size(0)
        valid_mask = (targets_flat != ignore_index) & (targets_flat >= 0) & (targets_flat < vocab_size)
        is_all_valid = bool(valid_mask.all().item()) if valid_mask.numel() > 0 else True

        if is_all_valid:
            valid_indices = None
            n_valid = targets_flat.numel()
        else:
            valid_indices = torch.nonzero(valid_mask).squeeze(-1)
            n_valid = valid_indices.numel()

        v_chunk_size = vocab_chunk_size if (vocab_chunk_size is not None and 0 < vocab_chunk_size < vocab_size) else vocab_size
        ctx.v_chunk_size = v_chunk_size
        ctx.vocab_size = vocab_size
        ctx.is_all_valid = is_all_valid

        lse_flat = torch.zeros(targets_flat.shape, dtype=torch.float32, device=hidden.device)

        inv_t = 1.0 / ctx.temperature
        if n_valid > 0:
            b_float = bias.float() if bias is not None else None
            w_compute_full = weight.to(compute_dtype) if v_chunk_size == vocab_size else None

            for start in range(0, n_valid, chunk_size):
                end = min(start + chunk_size, n_valid)
                idx_c = slice(start, end) if is_all_valid else valid_indices[start:end]
                h_c = hidden_flat[idx_c].to(compute_dtype)
                tgt_c = targets_flat[idx_c]

                if v_chunk_size == vocab_size:
                    logits_c = (h_c @ w_compute_full.t()).float()
                    if b_float is not None:
                        logits_c = logits_c + b_float
                    if ctx.temperature != 1.0:
                        logits_c = logits_c * inv_t

                    chunk_log_probs = log_softmax(logits_c, tgt_c)
                    log_probs_flat[idx_c] = chunk_log_probs
                    lse_flat[idx_c] = torch.logsumexp(logits_c, dim=-1)
                else:
                    m = torch.full((h_c.size(0),), float("-inf"), dtype=torch.float32, device=hidden.device)
                    s = torch.zeros((h_c.size(0),), dtype=torch.float32, device=hidden.device)
                    target_logits = torch.zeros((h_c.size(0),), dtype=torch.float32, device=hidden.device)

                    for v_start in range(0, vocab_size, v_chunk_size):
                        v_end = min(v_start + v_chunk_size, vocab_size)
                        w_v = weight[v_start:v_end].to(compute_dtype)
                        logits_v = (h_c @ w_v.t()).float()
                        if b_float is not None:
                            logits_v = logits_v + b_float[v_start:v_end]
                        if ctx.temperature != 1.0:
                            logits_v = logits_v * inv_t

                        in_chunk = (tgt_c >= v_start) & (tgt_c < v_end)
                        if in_chunk.any():
                            local_tgt = tgt_c[in_chunk] - v_start
                            target_logits[in_chunk] = logits_v[in_chunk, local_tgt]

                        chunk_max = logits_v.max(dim=-1).values
                        new_m = torch.maximum(m, chunk_max)
                        s = s * torch.exp(m - new_m) + torch.exp(logits_v - new_m.unsqueeze(-1)).sum(dim=-1)
                        m = new_m

                    lse_c = m + torch.log(s)
                    log_probs_flat[idx_c] = target_logits - lse_c
                    lse_flat[idx_c] = lse_c

        ctx.save_for_backward(hidden, weight, bias, targets, valid_indices, lse_flat)

        ctx.n_valid = n_valid
        ctx.orig_h_shape = orig_h_shape
        ctx.orig_targets_shape = orig_targets_shape
        return log_probs_flat.view(orig_targets_shape)

    @staticmethod
    def backward(ctx, grad_log_probs: Optional[torch.Tensor]):
        hidden, weight, bias, targets, valid_indices, lse_flat = ctx.saved_tensors
        n_valid = ctx.n_valid
        orig_h_shape = getattr(ctx, "orig_h_shape", hidden.shape)
        compute_dtype = getattr(ctx, "compute_dtype", hidden.dtype)
        is_all_valid = getattr(ctx, "is_all_valid", False)
        temperature = ctx.temperature
        inv_t = 1.0 / temperature
        v_chunk_size = ctx.v_chunk_size
        vocab_size = ctx.vocab_size

        needs_h_grad = ctx.needs_input_grad[0]
        needs_w_grad = ctx.needs_input_grad[1]
        needs_b_grad = bias is not None and ctx.needs_input_grad[2]

        chunk_size = ctx.chunk_size

        hidden_flat = hidden.reshape(-1, hidden.shape[-1]) if hidden.dim() == 3 else hidden
        targets_flat = targets.reshape(-1) if targets.dim() > 1 else targets

        with maybe_gather_lm_head_ctx(weight, bias):
            grad_hidden = torch.zeros(orig_h_shape, dtype=torch.float32, device=hidden.device) if needs_h_grad else None
            grad_hidden_flat = grad_hidden.view(-1, hidden.shape[-1]) if grad_hidden is not None else None
            grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if needs_w_grad else None
            grad_bias = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device) if needs_b_grad else None

            if grad_log_probs is not None and n_valid > 0:
                g_in = grad_log_probs.float()
                g_in_flat = g_in.reshape(-1) if g_in.dim() > 1 else g_in
                b_float = bias.float() if bias is not None else None
                w_compute_full = weight.to(compute_dtype) if v_chunk_size == vocab_size else None

                for start in range(0, n_valid, chunk_size):
                    end = min(start + chunk_size, n_valid)
                    idx_c = slice(start, end) if is_all_valid else valid_indices[start:end]
                    g_c = g_in_flat[idx_c]

                    if not g_c.any():
                        continue

                    h_c = hidden_flat[idx_c].to(compute_dtype)
                    tgt_c = targets_flat[idx_c]
                    lse_c = lse_flat[idx_c]

                    if v_chunk_size == vocab_size:
                        logits_c = (h_c @ w_compute_full.t()).float()
                        if b_float is not None:
                            logits_c = logits_c + b_float
                        if temperature != 1.0:
                            logits_c = logits_c * inv_t

                        d_logits = F.softmax(logits_c, dim=-1).neg_()
                        d_logits.scatter_add_(
                            dim=1,
                            index=tgt_c.unsqueeze(1),
                            src=torch.ones((tgt_c.shape[0], 1), dtype=d_logits.dtype, device=d_logits.device),
                        )
                        d_logits.mul_(g_c.unsqueeze(-1) * inv_t)

                        if needs_b_grad:
                            grad_bias.add_(d_logits.sum(dim=0))

                        d_logits_c = d_logits.to(compute_dtype)
                        if needs_h_grad:
                            if is_all_valid:
                                # slice(start, end) 返回视图 View，直接原地累加到 grad_hidden_flat
                                _addmm_fp32(grad_hidden_flat[start:end], d_logits_c, w_compute_full)
                            else:
                                # 花式索引返回 Copy，必须先累加到临时张量再通过 __setitem__ 写回
                                grad_h_c = torch.zeros((h_c.size(0), h_c.size(1)), dtype=torch.float32, device=hidden.device)
                                _addmm_fp32(grad_h_c, d_logits_c, w_compute_full)
                                grad_hidden_flat[idx_c] = grad_h_c
                        if needs_w_grad:
                            _addmm_fp32(grad_weight, d_logits_c.t(), h_c)
                    else:
                        grad_h_accum = (
                            grad_hidden_flat[start:end]
                            if (needs_h_grad and is_all_valid)
                            else (torch.zeros((h_c.size(0), h_c.size(1)), dtype=torch.float32, device=hidden.device) if needs_h_grad else None)
                        )
                        for v_start in range(0, vocab_size, v_chunk_size):
                            v_end = min(v_start + v_chunk_size, vocab_size)
                            w_v = weight[v_start:v_end].to(compute_dtype)
                            logits_v = (h_c @ w_v.t()).float()
                            if b_float is not None:
                                logits_v = logits_v + b_float[v_start:v_end]
                            if temperature != 1.0:
                                logits_v = logits_v * inv_t

                            d_logits_v = torch.exp(logits_v - lse_c.unsqueeze(-1)).neg_()
                            in_chunk = (tgt_c >= v_start) & (tgt_c < v_end)
                            if in_chunk.any():
                                in_chunk_indices = torch.nonzero(in_chunk).squeeze(-1)
                                local_tgt = tgt_c[in_chunk_indices] - v_start
                                d_logits_v[in_chunk_indices, local_tgt] += 1.0
                            d_logits_v.mul_(g_c.unsqueeze(-1) * inv_t)

                            if needs_b_grad:
                                grad_bias[v_start:v_end].add_(d_logits_v.sum(dim=0))

                            d_logits_v_c = d_logits_v.to(compute_dtype)
                            if needs_h_grad:
                                _addmm_fp32(grad_h_accum, d_logits_v_c, w_v)
                            if needs_w_grad:
                                _addmm_fp32(grad_weight[v_start:v_end], d_logits_v_c.t(), h_c)

                        if needs_h_grad and not is_all_valid:
                            grad_hidden_flat[idx_c] = grad_h_accum

        return (
            grad_hidden.to(hidden.dtype) if grad_hidden is not None else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None, None, None, None, None
        )


def chunked_log_probs(
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        targets: torch.Tensor,
        lm_head_bias: Optional[torch.Tensor] = None,
        chunk_size: int = 256,
        ignore_index: int = -100,
        temperature: float = 1.0,
        vocab_chunk_size: Optional[int] = 32768,
) -> torch.Tensor:
    orig_shape = targets.shape
    if hidden_states.dim() == 3:
        hidden = hidden_states.reshape(-1, hidden_states.size(-1))
        flat_targets = targets.reshape(-1)
    else:
        hidden = hidden_states
        flat_targets = targets

    with maybe_gather_lm_head_ctx(lm_head_weight, lm_head_bias):
        flat_log_probs = ChunkedLogProbFunction.apply(
            hidden,
            lm_head_weight,
            lm_head_bias,
            flat_targets,
            chunk_size,
            ignore_index,
            temperature,
            vocab_chunk_size,
        )

    return flat_log_probs.view(orig_shape)


class ChunkedLMLoss(nn.Module):
    def __init__(
            self,
            chunk_size: int = 256,
            ignore_index: int = -100,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index

    def forward(
            self,
            hidden_states: torch.Tensor,
            lm_head_weight: torch.Tensor,
            labels: torch.Tensor,
            lm_head_bias: Optional[torch.Tensor] = None,
            teacher_logits: Optional[torch.Tensor] = None,
            kd_coef: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        has_teacher = (teacher_logits is not None and kd_coef > 0.0)
        if hidden_states.dim() == 3:
            if labels.dim() == 2 and labels.size(1) == hidden_states.size(1):
                hidden = hidden_states[:, :-1, :]
                shift_labels = labels[:, 1:]
                shift_teacher_logits = teacher_logits[:, :-1, :] if has_teacher else None
            else:
                hidden = hidden_states
                shift_labels = labels
                shift_teacher_logits = teacher_logits if has_teacher else None

            if not has_teacher:
                # add_fused_lm_head：在进入 Function.apply 前先用 mask 过滤有效 Token，
                # 避免对包含大量 -100 (Prompt/Pad) 的 [B, S-1, H] 做全量 reshape 拷贝
                vocab_size = lm_head_weight.ds_shape[0] if hasattr(lm_head_weight, "ds_shape") else lm_head_weight.size(0)
                mask = (shift_labels != self.ignore_index) & (shift_labels >= 0) & (shift_labels < vocab_size)
                hidden_filtered = hidden[mask]
                labels_filtered = shift_labels[mask]
                with maybe_gather_lm_head_ctx(lm_head_weight, lm_head_bias):
                    total_loss, ce_loss = ChunkedCrossEntropyFunction.apply(
                        hidden_filtered,
                        lm_head_weight,
                        lm_head_bias,
                        labels_filtered,
                        None,
                        self.chunk_size,
                        self.ignore_index,
                        0.0
                    )
                # 若当前 rank 全为 -100 (mask.any() == False)，保持 total_loss 与主干 hidden 连通
                if not mask.any():
                    total_loss = total_loss + (hidden.sum() * 0.0).to(total_loss.dtype)
                return total_loss, ce_loss

            hidden = hidden.reshape(-1, hidden.size(-1))
            shift_labels = shift_labels.reshape(-1)
            shift_teacher_logits = shift_teacher_logits.reshape(-1, shift_teacher_logits.size(-1)) if shift_teacher_logits is not None else None
        else:
            hidden = hidden_states
            shift_labels = labels
            shift_teacher_logits = teacher_logits if has_teacher else None

        with maybe_gather_lm_head_ctx(lm_head_weight, lm_head_bias):
            return ChunkedCrossEntropyFunction.apply(
                hidden,
                lm_head_weight,
                lm_head_bias,
                shift_labels,
                shift_teacher_logits,
                self.chunk_size,
                self.ignore_index,
                kd_coef
            )


class LMLoss(nn.Module):
    """
    llm loss
    """
    def __init__(
            self,
            ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits shape (batch, seq_len, vocab_size)
        # labels shape (batch, seq_len)

        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # logits = shift_logits.reshape(-1, logits.shape[-1])
        # targets = shift_labels.reshape(-1)

        shift_labels = F.pad(labels[..., 1:], (0, 1), value=self.ignore_index)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = shift_labels.reshape(-1)
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return (logits.float() * 0.0).sum()

        ce_loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
        )

        return ce_loss


class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits[..., :-1, :].contiguous()
        teacher_logits = teacher_logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)

        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        safe_logprobs = torch.clamp(logprobs, min=-100.0)
        prod_probs = torch.where(teacher_probs > 0, teacher_probs * safe_logprobs, torch.zeros_like(safe_logprobs))

        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).float().view(-1)
        safe_x = torch.where(mask > 0, x, torch.zeros_like(x))
        distil_loss = -torch.sum(safe_x) / torch.sum(mask).clamp(min=1.0)

        return distil_loss


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable computation of log(1 - exp(x)) for x <= 0.
    Branches at -ln 2 ≈ -0.6931471805599453 to avoid catastrophic cancellation:
    - For x < -ln 2, exp(x) < 0.5, so 1 - exp(x) does not suffer cancellation; use log1p(-exp(x)).
    - For x >= -ln 2, exp(x) >= 0.5; use log(-expm1(x)) to preserve precision as expm1(x) avoids subtraction cancellation.
    """
    t = -0.6931471805599453
    return torch.where(x < t, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x)))


class DPOLoss(nn.Module):
    def __init__(
            self,
            beta: float,
            label_smoothing: float = 0.0,
            ipo: bool = False,
            loss_type: str = 'dpo',
            simpo_gamma: float = 0.5
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo
        self.loss_type = loss_type
        self.simpo_gamma = simpo_gamma

    def forward(
            self,
            policy_chosen_logps: torch.Tensor,
            policy_reject_logps: torch.Tensor,
            ref_chosen_logps: Optional[torch.Tensor] = None,
            ref_reject_logps: Optional[torch.Tensor] = None,
            policy_chosen_means: Optional[torch.Tensor] = None,
            policy_reject_means: Optional[torch.Tensor] = None,
            chosen_lengths: Optional[torch.Tensor] = None,
            reject_lengths: Optional[torch.Tensor] = None,
            ref_chosen_means: Optional[torch.Tensor] = None,
            ref_reject_means: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.loss_type == 'orpo':
            p_chosen = torch.clamp(policy_chosen_means.float(), max=-1e-7)
            log_odds_chosen = p_chosen - log1mexp(p_chosen)

            p_reject = torch.clamp(policy_reject_means.float(), max=-1e-7)
            log_odds_rejected = p_reject - log1mexp(p_reject)

            log_odds_ratio = log_odds_chosen - log_odds_rejected
            losses = self.beta * -F.logsigmoid(log_odds_ratio)
            if chosen_lengths is not None and reject_lengths is not None:
                valid_pair = (chosen_lengths > 0) & (reject_lengths > 0)
                losses = torch.where(valid_pair, losses, torch.zeros_like(losses))
                return losses.sum() / valid_pair.float().sum().clamp(min=1.0)
            return losses.mean()

        if self.loss_type == 'simpo':
            # SimPO: Simple Preference Optimization
            # Incorporates label_smoothing regularization
            logits = (policy_chosen_means.float() - policy_reject_means.float()) - (self.simpo_gamma / self.beta)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1.0 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            return losses.mean()

        policy_chosen_logps = policy_chosen_logps.float()
        policy_reject_logps = policy_reject_logps.float()
        ref_chosen_logps = ref_chosen_logps.float() if ref_chosen_logps is not None else torch.zeros_like(policy_chosen_logps)
        ref_reject_logps = ref_reject_logps.float() if ref_reject_logps is not None else torch.zeros_like(policy_reject_logps)

        pi_logratios = policy_chosen_logps - policy_reject_logps
        ref_logratios = ref_chosen_logps - ref_reject_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            # IPO uses sequence-level log-prob differences; in code these are token-summed over the completion,
            # which makes the squared loss scale quadratically with completion length.
            # We therefore normalize by the number of completion tokens (average per token) to prevent squared loss explosion.
            if chosen_lengths is not None and reject_lengths is not None:
                chosen_avg_score = (policy_chosen_logps - ref_chosen_logps) / chosen_lengths.float().clamp(min=1.0)
                reject_avg_score = (policy_reject_logps - ref_reject_logps) / reject_lengths.float().clamp(min=1.0)
                ipo_delta = chosen_avg_score - reject_avg_score
            elif (policy_chosen_means is not None and ref_chosen_means is not None and
                  policy_reject_means is not None and ref_reject_means is not None):
                chosen_avg_score = policy_chosen_means.float() - ref_chosen_means.float()
                reject_avg_score = policy_reject_means.float() - ref_reject_means.float()
                ipo_delta = chosen_avg_score - reject_avg_score
            else:
                ipo_delta = logits

            losses = (ipo_delta - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                    -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()

        # chosen_rewards = self.beta * (policy_chosen_probs - ref_chosen_probs).detach()
        # rejected_rewards = self.beta * (policy_reject_probs - ref_reject_probs).detach()

        return loss


class PPOLoss(nn.Module):
    """
    PPO (Proximal Policy Optimization) 损失函数。
    这个类统一计算 Actor 和 Value 的损失。
    """

    def __init__(
            self,
            clip_eps: float,
            vf_coef: float,
            huber_delta: float = 1.0,
            value_clip_eps: Optional[float] = None
    ):
        """
        初始化PPO损失函数。
        :param clip_eps: PPO策略裁剪范围的epsilon值。
        :param vf_coef: 价值函数损失的系数。
        :param huber_delta: Huber Loss 的阈值。
        :param value_clip_eps: 价值函数裁剪范围的epsilon值。若为 None 则默认使用 clip_eps；若 <= 0 则禁用价值裁剪。
        """
        super().__init__()
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.huber_delta = huber_delta
        self.value_clip_eps = value_clip_eps if value_clip_eps is not None else clip_eps

    def forward(
            self,
            log_probs: torch.Tensor,
            old_log_probs: torch.Tensor,
            values: torch.Tensor,
            old_values: torch.Tensor,
            returns: torch.Tensor,
            advantages: torch.Tensor,
            mask: torch.Tensor,
            value_mask: Optional[torch.Tensor] = None
    ):
        """
        计算PPO的总损失、Actor损失和Value损失。

        :param log_probs: 当前策略的log probabilities, 形状: [batch_size, seq_len]
        :param old_log_probs: 生成rollout时的旧策略的log probabilities, 形状: [batch_size, seq_len]
        :param values: 当前评论家模型输出的价值, 形状: [batch_size, seq_len]
        :param old_values: 生成rollout时的旧价值, 形状: [batch_size, seq_len]
        :param returns: GAE计算出的回报, 形状: [batch_size, seq_len]
        :param advantages: GAE计算出的优势, 形状: [batch_size, seq_len]
        :param mask: 掩码，只计算生成部分的损失, 形状: [batch_size, seq_len]
        :return: (总损失, Actor损失, Value损失, Mean NLL)
        """
        if value_mask is None:
            value_mask = mask

        log_probs = log_probs.float()
        old_log_probs = old_log_probs.float()
        values = values.float()
        old_values = old_values.float()
        returns = returns.float()
        advantages = advantages.float()
        mask = mask.float()
        value_mask = value_mask.float()

        # Value Loss (价值损失) with optional independent clipping
        if self.value_clip_eps is not None and self.value_clip_eps > 0:
            values_clipped = old_values + torch.clamp(values - old_values, -self.value_clip_eps, self.value_clip_eps)
            vf_loss_unclipped = F.smooth_l1_loss(values, returns, reduction='none', beta=self.huber_delta)
            vf_loss_clipped = F.smooth_l1_loss(values_clipped, returns, reduction='none', beta=self.huber_delta)
            value_loss = torch.max(vf_loss_unclipped, vf_loss_clipped)
        else:
            value_loss = F.smooth_l1_loss(values, returns, reduction='none', beta=self.huber_delta)

        # Apply mask and average (F.smooth_l1_loss 内部二次区间已包含 0.5 系数，无需外层再乘 0.5)
        value_loss = (value_loss * value_mask).sum() / value_mask.sum().clamp(min=1.0)
        value_loss = value_loss * self.vf_coef

        # Actor Loss (策略损失)
        # 计算新旧策略的概率比 r_t = exp(log_prob_new - log_prob_old)
        # ratio 形状: [batch_size, seq_len]
        ratio = torch.exp(log_probs - old_log_probs)

        # PPO裁剪替代目标（Clipped Surrogate Objective）
        # surr1 形状: [batch_size, seq_len]
        surr1 = ratio * advantages
        # surr2 形状: [batch_size, seq_len]
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

        # 取两者中较小的一个，并加负号（因为我们要最大化这个目标，所以最小化它的负值）
        # 我们只关心生成部分（由mask标记）的损失
        actor_loss = -torch.sum(torch.min(surr1, surr2) * mask) / torch.sum(mask).clamp(min=1.0)

        # 总损失
        total_loss = actor_loss + value_loss

        with torch.no_grad():
            # 计算近似KL散度
            logratios = log_probs - old_log_probs
            approx_kl = torch.sum(((torch.exp(logratios) - 1) - logratios) * mask) / mask.sum().clamp(min=1.0)

            # 计算裁剪比例：只有 A>0 且 r>1+eps，或者 A<0 且 r<1-eps 时才真正触发有效截断
            clipped = ((advantages > 0) & ratio.gt(1.0 + self.clip_eps)) | \
                      ((advantages < 0) & ratio.lt(1.0 - self.clip_eps))
            clip_frac = torch.sum(clipped.float() * mask) / mask.sum().clamp(min=1.0)

            mean_nll = -torch.sum(log_probs * mask) / mask.sum().clamp(min=1.0)

        return total_loss, actor_loss, value_loss, approx_kl, clip_frac, mean_nll


class GRPOLoss(nn.Module):
    def __init__(
            self,
            beta: float,
            clip_eps_low: float,
            clip_eps_high: Optional[float] = None,
            delta: Optional[float] = None,
            importance_sampling_level: str = 'token',
            loss_type: str = 'grpo',
            sapo_temperature_pos: float = 1.0,
            sapo_temperature_neg: float = 1.0,
            vespo_k_pos: float = 2.0,
            vespo_lambda_pos: float = 3.0,
            vespo_k_neg: float = 3.0,
            vespo_lambda_neg: float = 2.0,
    ):
        super().__init__()

        self.beta = beta
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high if clip_eps_high is not None else clip_eps_low
        self.delta = delta
        self.importance_sampling_level = importance_sampling_level
        self.loss_type = loss_type

        self.sapo_temperature_pos = sapo_temperature_pos
        self.sapo_temperature_neg = sapo_temperature_neg
        self.vespo_k_pos = vespo_k_pos
        self.vespo_lambda_pos = vespo_lambda_pos
        self.vespo_k_neg = vespo_k_neg
        self.vespo_lambda_neg = vespo_lambda_neg

    @staticmethod
    @torch.no_grad()
    def get_gamma_weights(
            advantages: torch.Tensor,
            log_ratio_per_token: torch.Tensor,
            mask: torch.Tensor,
            k_pos: float,
            lambda_pos: float,
            k_neg: float,
            lambda_neg: float,
    ) -> torch.Tensor:
        advantages = advantages.float()
        log_ratio_per_token = log_ratio_per_token.float()
        mask = mask.float()

        lower_clamp = math.log(1e-8)
        log_ratio_clamped = torch.clamp(log_ratio_per_token, -20.0, 20.0)
        seq_log_ratio = torch.sum(log_ratio_clamped * mask, dim=-1, keepdim=True)

        log_w_seq = torch.clamp(seq_log_ratio, lower_clamp, 20.0)
        w_seq = torch.exp(log_w_seq)

        # 确保 advantages 为序列级标量 (B, 1)，消除逐 token 或高维 advantages 造成的张量维度广播错误
        if advantages.dim() >= 2 and advantages.size(-1) > 1:
            mask_sum = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            adv_seq = (advantages * mask).sum(dim=-1, keepdim=True) / mask_sum
        elif advantages.dim() == 1:
            adv_seq = advantages.unsqueeze(-1)
        else:
            adv_seq = advantages

        is_nonneg_adv = adv_seq >= 0
        k_seq = torch.where(
            is_nonneg_adv,
            torch.tensor(k_pos, device=advantages.device),
            torch.tensor(k_neg, device=advantages.device)
        )
        lambda_seq = torch.where(
            is_nonneg_adv,
            torch.tensor(lambda_pos, device=advantages.device),
            torch.tensor(lambda_neg, device=advantages.device)
        ).clamp(min=1e-4)

        # log(φ(w)) = λ + k × log(w) - λ × w
        log_phi = lambda_seq + k_seq * log_w_seq - lambda_seq * w_seq
        phi_seq = torch.exp(log_phi).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

        return phi_seq  # (B, 1)

    def forward(
            self,
            log_probs: torch.Tensor,
            old_log_probs: torch.Tensor,
            ref_log_probs: torch.Tensor,
            completion_mask: torch.Tensor,
            advantages: torch.Tensor,
            completion_len: int,
            avg_tokens_per_micro_batch: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = log_probs.float()
        old_log_probs = old_log_probs.float()
        if ref_log_probs is not None:
            ref_log_probs = ref_log_probs.float()
        completion_mask = completion_mask.float()
        advantages = advantages.float()

        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)

        if self.beta != 0.0 and ref_log_probs is not None:
            per_token_kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        else:
            per_token_kl = None

        log_ratio = log_probs - old_log_probs
        if self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1, keepdim=True) / completion_mask.sum(-1, keepdim=True).clamp(min=1.0)
        else:
            log_importance_weights = log_ratio

        raw_coef = torch.exp(log_importance_weights)
        coef_1 = raw_coef

        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.clip_eps_high).detach()
            per_token_loss = -clamped_ratios * advantages * log_probs

        elif self.loss_type == "sapo":
            temperatures = torch.where(
                advantages > 0,
                torch.tensor(self.sapo_temperature_pos, device=advantages.device),
                torch.tensor(self.sapo_temperature_neg, device=advantages.device)
            )
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages

        elif self.loss_type == "vespo":
            phi_seq = self.get_gamma_weights(
                advantages=advantages,
                log_ratio_per_token=log_ratio,
                mask=completion_mask,
                k_pos=self.vespo_k_pos,
                lambda_pos=self.vespo_lambda_pos,
                k_neg=self.vespo_k_neg,
                lambda_neg=self.vespo_lambda_neg
            )
            per_token_loss = -phi_seq * advantages * log_probs

        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(raw_coef, 1 - self.clip_eps_low, 1 + self.clip_eps_high)
            if self.delta is not None:
                coef_1 = torch.clamp(raw_coef, max=self.delta)
            else:
                coef_1 = raw_coef

            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.beta != 0.0 and per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type in ["bnpo", "cispo", "dapo", "vespo"]:
            # 无论 token 级还是 sequence 级 IS，per_token_loss 均可能因 log_probs、per_token_kl 或 2D advantages 具有 [B, T] 维度，必须乘以 completion_mask 屏蔽 Padding
            if avg_tokens_per_micro_batch is not None and avg_tokens_per_micro_batch > 0:
                loss = (per_token_loss * completion_mask).sum() / avg_tokens_per_micro_batch
            else:
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            completion_len = max(completion_len, 1)
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * completion_len)
        elif self.loss_type == "luspo":
            loss = (per_token_loss * completion_mask).sum(dim=-1).mean()
        else:
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        with torch.no_grad():
            if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
                is_clipped = (
                    ((advantages > 0) & (raw_coef > 1 + self.clip_eps_high)) |
                    ((advantages < 0) & (raw_coef < 1 - self.clip_eps_low))
                )
            elif self.loss_type == "cispo":
                is_clipped = (raw_coef > self.clip_eps_high) & (advantages > 0)
            else:
                is_clipped = (
                    ((advantages > 0) & (raw_coef > 1 + self.clip_eps_high)) |
                    ((advantages < 0) & (raw_coef < 1 - self.clip_eps_low))
                )
            clip_frac = (is_clipped.float() * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

        return loss, clip_frac
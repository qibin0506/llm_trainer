from typing import Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from .partition_utils import maybe_gather_lm_head_ctx
from .utils import log_softmax


def _get_compute_dtype(hidden: torch.Tensor) -> torch.dtype:
    device_type = hidden.device.type
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return hidden.dtype


def _get_global_valid_token_denom(local_n_valid: torch.Tensor) -> torch.Tensor:
    """
    计算跨卡平均有效 Token 数作为归一化分母，消除多卡之间因有效回答长度不均或单卡全 -100 导致的梯度偏置与稀释。
    """
    denom = local_n_valid.to(dtype=torch.float32)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        global_n_valid = denom.clone()
        dist.all_reduce(global_n_valid, op=dist.ReduceOp.SUM)
        return global_n_valid.clamp(min=1.0) / float(dist.get_world_size())
    return denom.clamp(min=1.0)


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
    3. 显式记录 compute_dtype 并全程采用 FP32 累加 grad_hidden 与 grad_weight，无任何 .item()/.any() GPU-CPU 同步阻塞。
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
            loss_denom: Optional[torch.Tensor] = None,
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
        n_tokens = hidden_flat.size(0)
        valid_mask = (labels_flat != ignore_index) & (labels_flat >= 0) & (labels_flat < vocab_size)
        if loss_denom is not None:
            denom = loss_denom.to(device=hidden.device, dtype=torch.float32).clamp(min=1e-8)
        else:
            local_n_valid = valid_mask.sum()
            denom = _get_global_valid_token_denom(local_n_valid)

        total_loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
        ce_loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)

        if n_tokens > 0:
            w_compute = weight.to(compute_dtype)
            b_float = bias.float() if bias is not None else None

            for start in range(0, n_tokens, chunk_size):
                end = min(start + chunk_size, n_tokens)
                h_c = hidden_flat[start:end].to(compute_dtype)
                lbl_c = labels_flat[start:end]
                valid_c = valid_mask[start:end]
                safe_lbl_c = torch.where(valid_c, lbl_c, torch.full_like(lbl_c, ignore_index))

                logits_c = (h_c @ w_compute.t()).float()
                if b_float is not None:
                    logits_c = logits_c + b_float

                log_p_c = F.log_softmax(logits_c, dim=-1)
                chunk_ce = F.nll_loss(log_p_c, safe_lbl_c, ignore_index=ignore_index, reduction="sum")
                ce_loss_sum = ce_loss_sum + chunk_ce

                if has_teacher:
                    t_logits_c = t_logits_flat[start:end].to(device=hidden.device, dtype=torch.float32)
                    t_probs_c = F.softmax(t_logits_c, dim=-1)
                    safe_log_p = torch.clamp(log_p_c, min=-100.0)
                    valid_c_2d = valid_c.unsqueeze(-1)
                    prod = torch.where(
                        (t_probs_c > 0) & valid_c_2d,
                        t_probs_c * safe_log_p,
                        torch.zeros_like(safe_log_p),
                    )
                    chunk_kd = -prod.sum()
                    chunk_total = (1.0 - kd_coef) * chunk_ce + kd_coef * chunk_kd
                else:
                    chunk_total = chunk_ce

                total_loss_sum = total_loss_sum + chunk_total

        t_logits_cpu = None
        if has_teacher and n_tokens > 0:
            # 直接使用常规 CPU 内存卸载，避免每步动态调用 pin_memory=True 触发高昂的 cudaHostAlloc 同步阻塞
            t_logits_cpu = t_logits_flat.detach().to(device="cpu", non_blocking=False)

        total_loss = total_loss_sum / denom
        ce_loss = (ce_loss_sum / denom).detach()

        del t_logits_flat

        ctx.save_for_backward(hidden_flat, weight, bias, labels_flat, denom)
        ctx.teacher_logits_cpu = t_logits_cpu
        ctx.n_tokens = n_tokens
        ctx.vocab_size = vocab_size
        ctx.orig_h_shape = orig_h_shape
        ctx.hidden_dtype = hidden.dtype
        ctx.mark_non_differentiable(ce_loss)
        return total_loss, ce_loss

    @staticmethod
    def backward(ctx, grad_total_loss: Optional[torch.Tensor], grad_ce_loss: Optional[torch.Tensor] = None):
        hidden_flat, weight, bias, labels_flat, denom = ctx.saved_tensors
        t_logits_cpu = getattr(ctx, "teacher_logits_cpu", None)
        n_tokens = ctx.n_tokens
        vocab_size = ctx.vocab_size
        ignore_index = ctx.ignore_index
        orig_h_shape = getattr(ctx, "orig_h_shape", hidden_flat.shape)
        hidden_dtype = getattr(ctx, "hidden_dtype", hidden_flat.dtype)
        compute_dtype = getattr(ctx, "compute_dtype", hidden_dtype)

        needs_h_grad = ctx.needs_input_grad[0]
        needs_w_grad = ctx.needs_input_grad[1]
        needs_b_grad = bias is not None and ctx.needs_input_grad[2]

        chunk_size = ctx.chunk_size
        kd_coef = ctx.kd_coef
        has_teacher = (t_logits_cpu is not None) and (kd_coef > 0.0)

        with maybe_gather_lm_head_ctx(weight, bias):
            grad_hidden = torch.zeros(orig_h_shape, dtype=torch.float32, device=hidden_flat.device) if needs_h_grad else None
            grad_hidden_flat = grad_hidden.view(-1, orig_h_shape[-1]) if grad_hidden is not None else None
            grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if needs_w_grad else None
            grad_bias = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device) if needs_b_grad else None

            if grad_total_loss is not None and n_tokens > 0:
                g_out = grad_total_loss.float()
                base_scale = g_out / denom

                w_compute = weight.to(compute_dtype)
                b_float = bias.float() if bias is not None else None

                for start in range(0, n_tokens, chunk_size):
                    end = min(start + chunk_size, n_tokens)
                    h_c = hidden_flat[start:end].to(compute_dtype)
                    lbl_c = labels_flat[start:end]
                    valid_c = (lbl_c != ignore_index) & (lbl_c >= 0) & (lbl_c < vocab_size)
                    safe_lbl_c = lbl_c.clamp(min=0, max=vocab_size - 1)
                    scale_c = torch.where(valid_c, base_scale, torch.zeros_like(base_scale)).unsqueeze(-1)

                    logits_c = (h_c @ w_compute.t()).float()
                    if b_float is not None:
                        logits_c = logits_c + b_float

                    if has_teacher:
                        log_p_c = F.log_softmax(logits_c, dim=-1)
                        d_logits = torch.exp(log_p_c)
                        t_logits_c = t_logits_cpu[start:end].to(hidden_flat.device, non_blocking=False).float()
                        t_probs_c = F.softmax(t_logits_c, dim=-1)
                        del t_logits_c

                        active_mask = (t_probs_c > 0) & (log_p_c >= -100.0)
                        masked_t_probs = torch.where(active_mask, t_probs_c, torch.zeros_like(t_probs_c))
                        t_sum = masked_t_probs.sum(dim=-1, keepdim=True)
                        d_logits.mul_((1.0 - kd_coef) + kd_coef * t_sum)
                        if kd_coef < 1.0:
                            d_logits.scatter_add_(
                                dim=1,
                                index=safe_lbl_c.unsqueeze(1),
                                src=torch.full((safe_lbl_c.shape[0], 1), -(1.0 - kd_coef), dtype=d_logits.dtype, device=d_logits.device),
                            )
                        d_logits.sub_(masked_t_probs.mul_(kd_coef))
                    else:
                        d_logits = F.softmax(logits_c, dim=-1)
                        d_logits.scatter_add_(
                            dim=1,
                            index=safe_lbl_c.unsqueeze(1),
                            src=torch.full((safe_lbl_c.shape[0], 1), -1.0, dtype=d_logits.dtype, device=d_logits.device),
                        )

                    d_logits.mul_(scale_c)
                    if needs_b_grad:
                        grad_bias.add_(d_logits.sum(dim=0))

                    d_logits_c = d_logits.to(compute_dtype)
                    if needs_h_grad:
                        _addmm_fp32(grad_hidden_flat[start:end], d_logits_c, w_compute)
                    if needs_w_grad:
                        _addmm_fp32(grad_weight, d_logits_c.t(), h_c)

        if hasattr(ctx, "teacher_logits_cpu"):
            del ctx.teacher_logits_cpu

        return (
            grad_hidden.to(hidden_dtype) if grad_hidden is not None else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None, None, None, None, None, None
        )


class ChunkedLogProbFunction(torch.autograd.Function):
    """
    分块计算目标 token 的 log_probs，彻底避免物化 [B, S, V] 全量 Logits。
    - 统一捕获 compute_dtype 并在循环外一次性转换 weight，避免每个分块重复转换主权重；
    - 反向传播中 grad_hidden 与 grad_weight 均采用 FP32 缓冲区配合 _addmm_fp32 原地累加；
    - 全程使用纯张量掩码替代 .item() / .any() / nonzero()，消除所有 GPU-CPU 同步阻塞。
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
            vocab_chunk_size: Optional[int] = None,
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

        n_tokens = targets_flat.numel()
        log_probs_flat = torch.zeros(targets_flat.shape, dtype=torch.float32, device=hidden.device)
        lse_flat = torch.zeros(targets_flat.shape, dtype=torch.float32, device=hidden.device)

        vocab_size = weight.ds_shape[0] if hasattr(weight, "ds_shape") else weight.size(0)
        v_chunk_size = vocab_chunk_size if (vocab_chunk_size is not None and 0 < vocab_chunk_size < vocab_size) else vocab_size
        ctx.v_chunk_size = v_chunk_size
        ctx.vocab_size = vocab_size

        inv_t = 1.0 / ctx.temperature
        if n_tokens > 0:
            b_float = bias.float() if bias is not None else None
            w_compute_full = weight.to(compute_dtype)

            for start in range(0, n_tokens, chunk_size):
                end = min(start + chunk_size, n_tokens)
                h_c = hidden_flat[start:end].to(compute_dtype)
                tgt_c = targets_flat[start:end]
                valid_c = (tgt_c != ignore_index) & (tgt_c >= 0) & (tgt_c < vocab_size)
                safe_tgt_c = tgt_c.clamp(min=0, max=vocab_size - 1)

                if v_chunk_size == vocab_size:
                    logits_c = (h_c @ w_compute_full.t()).float()
                    if b_float is not None:
                        logits_c = logits_c + b_float
                    if ctx.temperature != 1.0:
                        logits_c = logits_c * inv_t

                    chunk_log_probs = log_softmax(logits_c, safe_tgt_c)
                    log_probs_flat[start:end] = torch.where(valid_c, chunk_log_probs, torch.zeros_like(chunk_log_probs))
                    lse_flat[start:end] = torch.logsumexp(logits_c, dim=-1)
                else:
                    m = torch.full((h_c.size(0),), float("-inf"), dtype=torch.float32, device=hidden.device)
                    s = torch.zeros((h_c.size(0),), dtype=torch.float32, device=hidden.device)
                    target_logits = torch.zeros((h_c.size(0),), dtype=torch.float32, device=hidden.device)

                    for v_start in range(0, vocab_size, v_chunk_size):
                        v_end = min(v_start + v_chunk_size, vocab_size)
                        w_v = w_compute_full[v_start:v_end]
                        logits_v = (h_c @ w_v.t()).float()
                        if b_float is not None:
                            logits_v = logits_v + b_float[v_start:v_end]
                        if ctx.temperature != 1.0:
                            logits_v = logits_v * inv_t

                        in_chunk = (safe_tgt_c >= v_start) & (safe_tgt_c < v_end)
                        local_tgt = (safe_tgt_c - v_start).clamp(min=0, max=(v_end - v_start - 1)).unsqueeze(-1)
                        gathered_logits = logits_v.gather(-1, local_tgt).squeeze(-1)
                        target_logits = torch.where(in_chunk, gathered_logits, target_logits)

                        chunk_max = logits_v.max(dim=-1).values
                        new_m = torch.maximum(m, chunk_max)
                        safe_new_m = torch.where(torch.isinf(new_m), torch.zeros_like(new_m), new_m)
                        rescale = torch.where(m == new_m, torch.ones_like(m), torch.exp(m - safe_new_m))
                        s = s * rescale + torch.exp(logits_v - safe_new_m.unsqueeze(-1)).sum(dim=-1)
                        m = new_m

                    lse_c = m + torch.log(s)
                    safe_lse_c = torch.where(torch.isinf(lse_c), torch.zeros_like(lse_c), lse_c)
                    chunk_log_probs = target_logits - safe_lse_c
                    log_probs_flat[start:end] = torch.where(valid_c, chunk_log_probs, torch.zeros_like(chunk_log_probs))
                    lse_flat[start:end] = lse_c

        ctx.save_for_backward(hidden_flat, weight, bias, targets_flat, lse_flat)

        ctx.n_tokens = n_tokens
        ctx.orig_h_shape = orig_h_shape
        ctx.orig_targets_shape = orig_targets_shape
        return log_probs_flat.view(orig_targets_shape)

    @staticmethod
    def backward(ctx, grad_log_probs: Optional[torch.Tensor]):
        hidden_flat, weight, bias, targets_flat, lse_flat = ctx.saved_tensors
        n_tokens = ctx.n_tokens
        ignore_index = ctx.ignore_index
        orig_h_shape = getattr(ctx, "orig_h_shape", hidden_flat.shape)
        compute_dtype = getattr(ctx, "compute_dtype", hidden_flat.dtype)
        temperature = ctx.temperature
        inv_t = 1.0 / temperature
        v_chunk_size = ctx.v_chunk_size
        vocab_size = ctx.vocab_size

        needs_h_grad = ctx.needs_input_grad[0]
        needs_w_grad = ctx.needs_input_grad[1]
        needs_b_grad = bias is not None and ctx.needs_input_grad[2]

        chunk_size = ctx.chunk_size

        with maybe_gather_lm_head_ctx(weight, bias):
            grad_hidden = torch.zeros(orig_h_shape, dtype=torch.float32, device=hidden_flat.device) if needs_h_grad else None
            grad_hidden_flat = grad_hidden.view(-1, hidden_flat.shape[-1]) if grad_hidden is not None else None
            grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if needs_w_grad else None
            grad_bias = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device) if needs_b_grad else None

            if grad_log_probs is not None and n_tokens > 0:
                g_in = grad_log_probs.float()
                g_in_flat = g_in.reshape(-1) if g_in.dim() > 1 else g_in
                b_float = bias.float() if bias is not None else None
                w_compute_full = weight.to(compute_dtype)

                for start in range(0, n_tokens, chunk_size):
                    end = min(start + chunk_size, n_tokens)
                    g_c = g_in_flat[start:end]
                    h_c = hidden_flat[start:end].to(compute_dtype)
                    tgt_c = targets_flat[start:end]
                    lse_c = lse_flat[start:end]

                    valid_c = (tgt_c != ignore_index) & (tgt_c >= 0) & (tgt_c < vocab_size)
                    safe_tgt_c = tgt_c.clamp(min=0, max=vocab_size - 1)
                    scale_c = torch.where(valid_c, g_c * inv_t, torch.zeros_like(g_c)).unsqueeze(-1)

                    if v_chunk_size == vocab_size:
                        logits_c = (h_c @ w_compute_full.t()).float()
                        if b_float is not None:
                            logits_c = logits_c + b_float
                        if temperature != 1.0:
                            logits_c = logits_c * inv_t

                        d_logits = F.softmax(logits_c, dim=-1).neg_()
                        d_logits.scatter_add_(
                            dim=1,
                            index=safe_tgt_c.unsqueeze(1),
                            src=torch.ones((safe_tgt_c.shape[0], 1), dtype=d_logits.dtype, device=d_logits.device),
                        )
                        d_logits.mul_(scale_c)

                        if needs_b_grad:
                            grad_bias.add_(d_logits.sum(dim=0))

                        d_logits_c = d_logits.to(compute_dtype)
                        if needs_h_grad:
                            _addmm_fp32(grad_hidden_flat[start:end], d_logits_c, w_compute_full)
                        if needs_w_grad:
                            _addmm_fp32(grad_weight, d_logits_c.t(), h_c)
                    else:
                        grad_h_accum = grad_hidden_flat[start:end] if needs_h_grad else None
                        safe_lse_c = torch.where(torch.isinf(lse_c), torch.zeros_like(lse_c), lse_c)

                        for v_start in range(0, vocab_size, v_chunk_size):
                            v_end = min(v_start + v_chunk_size, vocab_size)
                            w_v = w_compute_full[v_start:v_end]
                            logits_v = (h_c @ w_v.t()).float()
                            if b_float is not None:
                                logits_v = logits_v + b_float[v_start:v_end]
                            if temperature != 1.0:
                                logits_v = logits_v * inv_t

                            d_logits_v = torch.exp(logits_v - safe_lse_c.unsqueeze(-1)).neg_()
                            in_chunk = (safe_tgt_c >= v_start) & (safe_tgt_c < v_end) & valid_c
                            local_tgt = (safe_tgt_c - v_start).clamp(min=0, max=(v_end - v_start - 1)).unsqueeze(1)
                            d_logits_v.scatter_add_(
                                dim=1,
                                index=local_tgt,
                                src=in_chunk.unsqueeze(1).to(d_logits_v.dtype),
                            )
                            d_logits_v.mul_(scale_c)

                            if needs_b_grad:
                                grad_bias[v_start:v_end].add_(d_logits_v.sum(dim=0))

                            d_logits_v_c = d_logits_v.to(compute_dtype)
                            if needs_h_grad:
                                _addmm_fp32(grad_h_accum, d_logits_v_c, w_v)
                            if needs_w_grad:
                                _addmm_fp32(grad_weight[v_start:v_end], d_logits_v_c.t(), h_c)

        return (
            grad_hidden.to(hidden_flat.dtype) if grad_hidden is not None else None,
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
        vocab_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    vocab_size = lm_head_weight.ds_shape[0] if hasattr(lm_head_weight, "ds_shape") else lm_head_weight.size(0)
    mask = (targets != ignore_index) & (targets >= 0) & (targets < vocab_size)

    with maybe_gather_lm_head_ctx(lm_head_weight, lm_head_bias):
        valid_log_probs = ChunkedLogProbFunction.apply(
            hidden_states[mask],
            lm_head_weight,
            lm_head_bias,
            targets[mask],
            chunk_size,
            ignore_index,
            temperature,
            vocab_chunk_size,
        )

    return valid_log_probs.new_zeros(targets.shape).masked_scatter(mask, valid_log_probs)


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
            loss_denom: Optional[torch.Tensor] = None,
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
        else:
            hidden = hidden_states
            shift_labels = labels
            shift_teacher_logits = teacher_logits if has_teacher else None

        # 无论是否开启知识蒸馏 (has_teacher)，统一先通过布尔掩码过滤有效 Token，
        # 避免对非连续切片 [B, S-1, V] 调用 .reshape() 触发全量显存深拷贝
        vocab_size = lm_head_weight.ds_shape[0] if hasattr(lm_head_weight, "ds_shape") else lm_head_weight.size(0)
        mask = (shift_labels != self.ignore_index) & (shift_labels >= 0) & (shift_labels < vocab_size)
        hidden_filtered = hidden[mask]
        labels_filtered = shift_labels[mask]
        teacher_filtered = shift_teacher_logits[mask] if shift_teacher_logits is not None else None

        with maybe_gather_lm_head_ctx(lm_head_weight, lm_head_bias):
            total_loss, ce_loss = ChunkedCrossEntropyFunction.apply(
                hidden_filtered,
                lm_head_weight,
                lm_head_bias,
                labels_filtered,
                teacher_filtered,
                self.chunk_size,
                self.ignore_index,
                kd_coef if has_teacher else 0.0,
                loss_denom,
            )

        # 始终保持 total_loss 与主干 hidden 的 Autograd 梯度图连通，防止全 -100 微批次下梯度断连
        total_loss = total_loss + (hidden.sum() * 0.0).to(total_loss.dtype)
        return total_loss, ce_loss


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

    def forward(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            loss_denom: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_labels = F.pad(labels[..., 1:], (0, 1), value=self.ignore_index)
        logits = logits.reshape(-1, logits.shape[-1])
        targets = shift_labels.reshape(-1)
        vocab_size = logits.shape[-1]
        valid_mask = (targets != self.ignore_index) & (targets >= 0) & (targets < vocab_size)
        safe_targets = torch.where(valid_mask, targets, torch.full_like(targets, self.ignore_index))
        if loss_denom is not None:
            denom = loss_denom.to(device=logits.device, dtype=torch.float32).clamp(min=1e-8)
        else:
            denom = _get_global_valid_token_denom(valid_mask.sum())

        ce_loss_sum = F.cross_entropy(
            logits,
            safe_targets,
            ignore_index=self.ignore_index,
            reduction="sum",
        )
        return ce_loss_sum / denom + (logits.sum() * 0.0).to(ce_loss_sum.dtype)


class ChunkedKDLossFunction(torch.autograd.Function):
    """
    分块计算知识蒸馏损失 (Knowledge Distillation Loss)，避免在 [B, S, V] 全量张量上同时物化
    teacher_probs、logprobs、safe_logprobs、prod_probs 多份 FP32 巨型张量（如 152K 词表下瞬间占用 30+ GB 显存），
    同时省去对非连续切片 [..., :-1, :] 调用 .contiguous() 带来的全量显存深拷贝。
    """
    @staticmethod
    def forward(
            ctx,
            logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            labels: torch.Tensor,
            ignore_index: int = -100,
            chunk_size: int = 512,
            loss_denom: Optional[torch.Tensor] = None,
            shift_labels: bool = True,
    ) -> torch.Tensor:
        ctx.set_materialize_grads(False)
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        ctx.shift_labels = shift_labels

        if shift_labels:
            if logits.dim() == 3:
                shift_logits = logits[:, :-1, :]
                shift_teacher = teacher_logits[:, :-1, :]
                shift_labels_t = labels[:, 1:]
            elif logits.dim() == 2:
                # 2D 未切片单序列 [S, V] 与 [S]：执行时序错位切片 [:-1, :] 与 [1:]
                shift_logits = logits[:-1, :].unsqueeze(0)
                shift_teacher = teacher_logits[:-1, :].unsqueeze(0)
                shift_labels_t = labels[1:].unsqueeze(0)
            else:
                raise ValueError(f"ChunkedKDLossFunction only supports 2D or 3D logits, got {logits.dim()}D")
        else:
            if logits.dim() == 3:
                shift_logits = logits
                shift_teacher = teacher_logits
                shift_labels_t = labels
            elif logits.dim() == 2:
                shift_logits = logits.unsqueeze(0)
                shift_teacher = teacher_logits.unsqueeze(0)
                shift_labels_t = labels.unsqueeze(0)
            else:
                raise ValueError(f"ChunkedKDLossFunction only supports 2D or 3D logits, got {logits.dim()}D")

        vocab_size = shift_logits.size(-1)
        valid_mask = (shift_labels_t != ignore_index) & (shift_labels_t >= 0) & (shift_labels_t < vocab_size)
        if loss_denom is not None:
            denom = loss_denom.to(device=logits.device, dtype=torch.float32).clamp(min=1e-8)
        else:
            denom = _get_global_valid_token_denom(valid_mask.sum())

        loss_sum = torch.zeros((), dtype=torch.float32, device=logits.device)
        batch_size, seq_len, _ = shift_logits.shape

        if batch_size > 0 and seq_len > 0:
            for b in range(batch_size):
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    valid_c = valid_mask[b, start:end].unsqueeze(-1)
                    l_c = shift_logits[b, start:end, :].float()
                    t_c = shift_teacher[b, start:end, :].to(device=logits.device, dtype=torch.float32)

                    t_probs_c = F.softmax(t_c, dim=-1)
                    log_p_c = F.log_softmax(l_c, dim=-1)
                    safe_log_p = torch.clamp(log_p_c, min=-100.0)
                    prod = torch.where(
                        (t_probs_c > 0) & valid_c,
                        t_probs_c * safe_log_p,
                        torch.zeros_like(safe_log_p),
                    )
                    loss_sum = loss_sum - prod.sum()

        ctx.save_for_backward(logits, teacher_logits, labels, denom)
        return loss_sum / denom

    @staticmethod
    def backward(ctx, grad_output: Optional[torch.Tensor]):
        logits, teacher_logits, labels, denom = ctx.saved_tensors
        ignore_index = ctx.ignore_index
        chunk_size = ctx.chunk_size
        shift_labels = ctx.shift_labels

        needs_l_grad = ctx.needs_input_grad[0]
        needs_t_grad = ctx.needs_input_grad[1]
        grad_logits = torch.zeros_like(logits) if needs_l_grad else None
        grad_teacher = torch.zeros_like(teacher_logits) if needs_t_grad else None

        if grad_output is not None and (needs_l_grad or needs_t_grad):
            if shift_labels:
                if logits.dim() == 3:
                    shift_logits = logits[:, :-1, :]
                    shift_teacher = teacher_logits[:, :-1, :]
                    shift_labels_t = labels[:, 1:]
                    grad_shift_l = grad_logits[:, :-1, :] if grad_logits is not None else None
                    grad_shift_t = grad_teacher[:, :-1, :] if grad_teacher is not None else None
                else:
                    shift_logits = logits[:-1, :].unsqueeze(0)
                    shift_teacher = teacher_logits[:-1, :].unsqueeze(0)
                    shift_labels_t = labels[1:].unsqueeze(0)
                    grad_shift_l = grad_logits[:-1, :].unsqueeze(0) if grad_logits is not None else None
                    grad_shift_t = grad_teacher[:-1, :].unsqueeze(0) if grad_teacher is not None else None
            else:
                if logits.dim() == 3:
                    shift_logits = logits
                    shift_teacher = teacher_logits
                    shift_labels_t = labels
                    grad_shift_l = grad_logits if grad_logits is not None else None
                    grad_shift_t = grad_teacher if grad_teacher is not None else None
                else:
                    shift_logits = logits.unsqueeze(0)
                    shift_teacher = teacher_logits.unsqueeze(0)
                    shift_labels_t = labels.unsqueeze(0)
                    grad_shift_l = grad_logits.unsqueeze(0) if grad_logits is not None else None
                    grad_shift_t = grad_teacher.unsqueeze(0) if grad_teacher is not None else None

            vocab_size = shift_logits.size(-1)
            batch_size, seq_len, _ = shift_logits.shape
            base_scale = grad_output.float() / denom

            if batch_size > 0 and seq_len > 0:
                for b in range(batch_size):
                    for start in range(0, seq_len, chunk_size):
                        end = min(start + chunk_size, seq_len)
                        lbl_c = shift_labels_t[b, start:end]
                        valid_c = (lbl_c != ignore_index) & (lbl_c >= 0) & (lbl_c < vocab_size)
                        scale_c = torch.where(valid_c, base_scale, torch.zeros_like(base_scale)).unsqueeze(-1)

                        l_c = shift_logits[b, start:end, :].float()
                        t_c = shift_teacher[b, start:end, :].to(device=logits.device, dtype=torch.float32)

                        t_probs_c = F.softmax(t_c, dim=-1)
                        log_p_c = F.log_softmax(l_c, dim=-1)
                        p_s_c = torch.exp(log_p_c)

                        active_mask = (t_probs_c > 0) & (log_p_c >= -100.0)
                        t_probs_eff = torch.where(active_mask, t_probs_c, torch.zeros_like(t_probs_c))

                        if grad_shift_l is not None:
                            t_sum = t_probs_eff.sum(dim=-1, keepdim=True)
                            d_l_c = (p_s_c * t_sum - t_probs_eff) * scale_c
                            grad_shift_l[b, start:end, :].copy_(d_l_c.to(logits.dtype))

                        if grad_shift_t is not None:
                            safe_log_p = torch.where(t_probs_c > 0, torch.clamp(log_p_c, min=-100.0), torch.zeros_like(log_p_c))
                            expected_log_p = (t_probs_c * safe_log_p).sum(dim=-1, keepdim=True)
                            d_t_c = -t_probs_c * (safe_log_p - expected_log_p) * scale_c
                            grad_shift_t[b, start:end, :].copy_(d_t_c.to(teacher_logits.dtype))

        return grad_logits, grad_teacher, None, None, None, None, None


class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """
    def __init__(self, ignore_index: int = -100, chunk_size: int = 512, shift_labels: bool = True):
        super().__init__()
        self.ignore_index = ignore_index
        self.chunk_size = chunk_size
        self.shift_labels = shift_labels

    def forward(
            self,
            logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            labels: torch.Tensor,
            loss_denom: Optional[torch.Tensor] = None,
            shift_labels: Optional[bool] = None,
    ) -> torch.Tensor:
        if shift_labels is None:
            shift_labels = self.shift_labels
        distil_loss = ChunkedKDLossFunction.apply(
            logits,
            teacher_logits,
            labels,
            self.ignore_index,
            self.chunk_size,
            loss_denom,
            shift_labels,
        )
        return distil_loss + (logits.sum() * 0.0).to(torch.float32)


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

    @staticmethod
    def _masked_mean(
            losses: torch.Tensor,
            chosen_lengths: Optional[torch.Tensor],
            reject_lengths: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if chosen_lengths is not None and reject_lengths is not None:
            valid_pair = (chosen_lengths > 0) & (reject_lengths > 0)
            valid_losses = torch.where(valid_pair, losses, torch.zeros_like(losses))
            return valid_losses.sum() / valid_pair.float().sum().clamp(min=1.0) + 0.0 * losses.sum()
        return losses.mean()

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
            return self._masked_mean(losses, chosen_lengths, reject_lengths)

        if self.loss_type == 'simpo':
            # SimPO: Simple Preference Optimization
            # Incorporates label_smoothing regularization
            logits = (policy_chosen_means.float() - policy_reject_means.float()) - (self.simpo_gamma / self.beta)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1.0 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
            return self._masked_mean(losses, chosen_lengths, reject_lengths)

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

        return self._masked_mean(losses, chosen_lengths, reject_lengths)


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
            value_clip_eps: Optional[float] = None,
            value_loss_type: str = 'mse',
    ):
        """
        初始化PPO损失函数。
        :param clip_eps: PPO策略裁剪范围的epsilon值。
        :param vf_coef: 价值函数损失的系数。
        :param huber_delta: Huber Loss 的阈值（当 value_loss_type='huber' 时生效）。
        :param value_clip_eps: 价值函数裁剪范围的epsilon值。若为 None 则默认使用 clip_eps；若 <= 0 则禁用价值裁剪。
        :param value_loss_type: 价值函数损失类型，支持 'mse' 或 'huber'。
        """
        super().__init__()
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.huber_delta = huber_delta
        self.value_clip_eps = value_clip_eps if value_clip_eps is not None else clip_eps
        self.value_loss_type = value_loss_type

    def _compute_elementwise_vf_loss(self, pred_values: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        if self.value_loss_type == 'huber':
            # F.smooth_l1_loss 内部在 |x - y| < beta 二次区间已包含 0.5 系数
            return F.smooth_l1_loss(pred_values, target_returns, reduction='none', beta=self.huber_delta)
        # 标准 MSE Value Loss: 0.5 * (V - R)^2，与 Huber 二次区间的 0.5 系数严格对齐
        return 0.5 * ((pred_values - target_returns) ** 2)

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
            vf_loss_unclipped = self._compute_elementwise_vf_loss(values, returns)
            vf_loss_clipped = self._compute_elementwise_vf_loss(values_clipped, returns)
            value_loss = torch.max(vf_loss_unclipped, vf_loss_clipped)
        else:
            value_loss = self._compute_elementwise_vf_loss(values, returns)

        # Apply mask and average (MSE 与 Huber 内部均已包含 0.5 系数，外层无需重复乘 0.5)
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
            use_bias_correction_kl: bool = True,
    ):
        super().__init__()

        self.beta = beta
        self.clip_eps_low = clip_eps_low
        if loss_type == "cispo":
            # CISPO 中 clip_eps_high 代表重要性采样比率的绝对截断上界 (ScaleRL 推荐值为 5.0)
            # 若未显式传入或沿用了传统 GRPO 默认的相对偏移量 (<= 1.0)，自动重设为 5.0
            if clip_eps_high is None or clip_eps_high <= 1.0:
                self.clip_eps_high = 5.0
            else:
                self.clip_eps_high = clip_eps_high
        else:
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
        self.use_bias_correction_kl = use_bias_correction_kl

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

        log_ratio = log_probs - old_log_probs
        if self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1, keepdim=True) / completion_mask.sum(-1, keepdim=True).clamp(min=1.0)
        else:
            log_importance_weights = log_ratio

        raw_coef = torch.exp(log_importance_weights)
        coef_1 = raw_coef

        if self.beta != 0.0 and ref_log_probs is not None:
            per_token_kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
            if self.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1.detach()
        else:
            per_token_kl = None

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
                if self.delta is not None:
                    is_clipped = is_clipped | ((advantages != 0) & (raw_coef > self.delta))
            elif self.loss_type == "cispo":
                is_clipped = (raw_coef > self.clip_eps_high) & (advantages > 0)
            else:
                is_clipped = (
                    ((advantages > 0) & (raw_coef > 1 + self.clip_eps_high)) |
                    ((advantages < 0) & (raw_coef < 1 - self.clip_eps_low))
                )
            clip_frac = (is_clipped.float() * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

        return loss, clip_frac
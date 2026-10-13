from typing import Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
from .partition_utils import maybe_gather_lm_head_ctx
from .utils import log_softmax


class ChunkedCrossEntropyFunction(torch.autograd.Function):
    """
    1. 前向：由外部调用模块（如 ChunkedLMLoss）统一包裹 maybe_gather_lm_head_ctx 保护 .apply() 调用，
       算子内部直接执行分块 GEMM，累加标量损失，不保存任何中间 Logits，规避多层嵌套破坏 ZeRO-3 状态机。
    2. 反向：在 backward 内部统一进入一次 maybe_gather_lm_head_ctx，保证分块重算与 GEMM 累积期间权重处于 AVAILABLE 状态且全卡对齐。
    3. 彻底规避 torch.utils.checkpoint 的二次通信与反向访问已释放参数 (Data Hazard) 缺陷。
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
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.kd_coef = kd_coef
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

        valid_mask = labels_flat != ignore_index
        valid_indices = torch.nonzero(valid_mask).squeeze(-1)
        n_valid = valid_indices.numel()

        total_loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
        ce_loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)

        # 方案优化：绝不将全量 [B, S, V] 或任何 GPU teacher_logits 存入 ctx.save_for_backward！
        # 仅过滤有效 Token (valid_indices) 的 teacher_logits，并在需要时 offload 到 CPU (pinned memory)。
        t_logits_cpu = None
        if has_teacher and n_valid > 0:
            t_logits_valid = t_logits_flat[valid_indices]
            if t_logits_valid.is_cuda:
                t_logits_cpu = t_logits_valid.to("cpu", non_blocking=True)
                if torch.cuda.is_available():
                    try:
                        t_logits_cpu = t_logits_cpu.pin_memory()
                    except Exception:
                        pass
            else:
                t_logits_cpu = t_logits_valid
        else:
            t_logits_valid = None

        if n_valid > 0:
            for start in range(0, n_valid, chunk_size):
                idx_c = valid_indices[start : start + chunk_size]
                h_c = hidden_flat[idx_c]
                lbl_c = labels_flat[idx_c]

                logits_c = (h_c @ weight.t()).float()
                if bias is not None:
                    logits_c = logits_c + bias.float()

                log_p_c = F.log_softmax(logits_c, dim=-1)
                chunk_ce = F.nll_loss(log_p_c, lbl_c, ignore_index=ignore_index, reduction="sum")
                ce_loss_sum = ce_loss_sum + chunk_ce

                if has_teacher:
                    if t_logits_valid.device.type == hidden.device.type:
                        t_logits_c = t_logits_valid[start : start + chunk_size].float()
                    else:
                        t_logits_c = t_logits_valid[start : start + chunk_size].to(hidden.device, non_blocking=True).float()

                    t_probs_c = F.softmax(t_logits_c, dim=-1)
                    safe_log_p = torch.clamp(log_p_c, min=-100.0)
                    prod = torch.where(t_probs_c > 0, t_probs_c * safe_log_p, torch.zeros_like(safe_log_p))
                    chunk_kd = -prod.sum()
                    chunk_total = (1.0 - kd_coef) * chunk_ce + kd_coef * chunk_kd
                else:
                    chunk_total = chunk_ce

                total_loss_sum = total_loss_sum + chunk_total

        denom = max(n_valid, 1)
        total_loss = total_loss_sum / denom
        ce_loss = (ce_loss_sum / denom).detach()

        # 显式解除局部 GPU 临时张量引用
        del t_logits_valid, t_logits_flat

        ctx.save_for_backward(hidden, weight, bias, labels, valid_indices)

        ctx.teacher_logits_cpu = t_logits_cpu
        ctx.n_valid = n_valid
        ctx.orig_h_shape = orig_h_shape
        return total_loss, ce_loss

    @staticmethod
    def backward(ctx, grad_total_loss: torch.Tensor, grad_ce_loss: Optional[torch.Tensor]):
        hidden, weight, bias, labels, valid_indices = ctx.saved_tensors
        t_logits_cpu = getattr(ctx, "teacher_logits_cpu", None)
        n_valid = ctx.n_valid
        orig_h_shape = getattr(ctx, "orig_h_shape", hidden.shape)

        needs_h_grad = ctx.needs_input_grad[0]
        needs_w_grad = ctx.needs_input_grad[1]
        needs_b_grad = bias is not None and ctx.needs_input_grad[2]

        chunk_size = ctx.chunk_size
        kd_coef = ctx.kd_coef
        has_teacher = (t_logits_cpu is not None) and (kd_coef > 0.0)

        g_out = grad_total_loss if grad_total_loss is not None else torch.zeros((), device=hidden.device)
        scale = g_out / max(n_valid, 1)

        grad_hidden = torch.zeros(orig_h_shape, dtype=hidden.dtype, device=hidden.device) if needs_h_grad else None
        grad_hidden_flat = grad_hidden.view(-1, hidden.shape[-1]) if grad_hidden is not None else None

        hidden_flat = hidden.reshape(-1, hidden.shape[-1]) if hidden.dim() == 3 else hidden
        labels_flat = labels.reshape(-1) if labels.dim() > 1 else labels

        # 整个反向重算循环进入 maybe_gather_lm_head_ctx，保证 weight/bias 处于 AVAILABLE 状态且所有 rank 集合通信严格对齐
        # 严禁在 with 外面基于 n_valid == 0 提前 return，避免 NCCL AllGather 死锁！
        with maybe_gather_lm_head_ctx(weight, bias):
            vocab_size = weight.ds_shape[0] if hasattr(weight, "ds_shape") else weight.shape[0]
            grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if needs_w_grad else None
            grad_bias = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device) if needs_b_grad else None

            if n_valid > 0:
                for start in range(0, n_valid, chunk_size):
                    idx_c = valid_indices[start : start + chunk_size]
                    h_c = hidden_flat[idx_c]
                    lbl_c = labels_flat[idx_c]

                    logits_c = (h_c @ weight.t()).float()
                    if bias is not None:
                        logits_c = logits_c + bias.float()

                    d_logits = F.softmax(logits_c, dim=-1)

                    if has_teacher:
                        # 按需从 CPU 搬运当前 chunk 的切片到计算设备，避免全量 logits 常驻 GPU 显存
                        t_logits_c = t_logits_cpu[start : start + chunk_size].to(hidden.device, non_blocking=True).float()
                        t_probs_c = F.softmax(t_logits_c, dim=-1)
                        # target_dist = (1 - kd_coef) * target_one_hot + kd_coef * t_probs_c
                        # d_logits = probs - target_dist = probs - kd_coef * t_probs_c - (1 - kd_coef) * target_one_hot
                        # 原地扣减蒸馏项与目标位置，彻底避免物化 [chunk_size, V] 的 Dense One-Hot 张量
                        d_logits.sub_(t_probs_c.mul_(kd_coef))
                        d_logits.scatter_add_(
                            dim=1,
                            index=lbl_c.unsqueeze(1),
                            src=torch.full((lbl_c.shape[0], 1), -(1.0 - kd_coef), dtype=d_logits.dtype, device=d_logits.device),
                        )
                    else:
                        # d_logits = probs - target_one_hot，直接通过 scatter_add_ 原地扣减目标位置的 1.0
                        d_logits.scatter_add_(
                            dim=1,
                            index=lbl_c.unsqueeze(1),
                            src=torch.full((lbl_c.shape[0], 1), -1.0, dtype=d_logits.dtype, device=d_logits.device),
                        )

                    d_logits.mul_(scale)
                    d_logits_compute = d_logits.to(h_c.dtype)

                    if needs_h_grad:
                        grad_hidden_flat[idx_c] = d_logits_compute @ weight
                    if needs_w_grad:
                        grad_weight.add_(d_logits.t() @ h_c.float())
                    if needs_b_grad:
                        grad_bias.add_(d_logits.sum(dim=0))

        # 反向计算完成后立即解除 CPU 引用，加快 Host 内存释放
        if hasattr(ctx, "teacher_logits_cpu"):
            del ctx.teacher_logits_cpu

        return (
            grad_hidden,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None, None, None, None, None
        )


class ChunkedLogProbFunction(torch.autograd.Function):
    """
    分块计算目标 token 的 log_probs，彻底避免物化 [B, S, V] 全量 Logits：
    1. 前向：由外部调用函数（如 chunked_log_probs）统一包裹 maybe_gather_lm_head_ctx 保护 .apply() 调用，
       算子内部直接执行分块 GEMM + log_softmax，仅保留目标 Token 的标量对数概率。
    2. 反向：接收上游梯度 grad_log_probs，在 backward 内部统一进入一次 maybe_gather_lm_head_ctx，
       分块重算并反向累积 hidden_states 与 lm_head.weight 梯度。
    3. 全流程闭环兼容 DeepSpeed ZeRO-3 状态机，杜绝多层上下文嵌套带来的生命周期冲突。
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
    ) -> torch.Tensor:
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index

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
        valid_indices = torch.nonzero(valid_mask).squeeze(-1)
        n_valid = valid_indices.numel()

        if n_valid > 0:
            for start in range(0, n_valid, chunk_size):
                idx_c = valid_indices[start : start + chunk_size]
                h_c = hidden_flat[idx_c]
                tgt_c = targets_flat[idx_c]

                logits_c = (h_c @ weight.t()).float()
                if bias is not None:
                    logits_c = logits_c + bias.float()

                chunk_log_probs = log_softmax(logits_c, tgt_c)
                log_probs_flat[idx_c] = chunk_log_probs

        ctx.save_for_backward(hidden, weight, bias, targets, valid_indices)

        ctx.n_valid = n_valid
        ctx.orig_h_shape = orig_h_shape
        ctx.orig_targets_shape = orig_targets_shape
        return log_probs_flat.view(orig_targets_shape).to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_log_probs: torch.Tensor):
        hidden, weight, bias, targets, valid_indices = ctx.saved_tensors
        n_valid = ctx.n_valid
        orig_h_shape = getattr(ctx, "orig_h_shape", hidden.shape)

        needs_h_grad = ctx.needs_input_grad[0]
        needs_w_grad = ctx.needs_input_grad[1]
        needs_b_grad = bias is not None and ctx.needs_input_grad[2]

        chunk_size = ctx.chunk_size
        g_in = grad_log_probs.float() if grad_log_probs is not None else None
        g_in_flat = g_in.reshape(-1) if (g_in is not None and g_in.dim() > 1) else g_in

        grad_hidden = torch.zeros(orig_h_shape, dtype=hidden.dtype, device=hidden.device) if needs_h_grad else None
        grad_hidden_flat = grad_hidden.view(-1, hidden.shape[-1]) if grad_hidden is not None else None

        hidden_flat = hidden.reshape(-1, hidden.shape[-1]) if hidden.dim() == 3 else hidden
        targets_flat = targets.reshape(-1) if targets.dim() > 1 else targets

        # 整个反向重算循环进入 maybe_gather_lm_head_ctx，保证 weight/bias 处于 AVAILABLE 状态且 rank 通信对齐
        with maybe_gather_lm_head_ctx(weight, bias):
            vocab_size = weight.ds_shape[0] if hasattr(weight, "ds_shape") else weight.size(0)
            grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device) if needs_w_grad else None
            grad_bias = torch.zeros(bias.shape, dtype=torch.float32, device=bias.device) if needs_b_grad else None

            if n_valid > 0 and g_in_flat is not None:
                for start in range(0, n_valid, chunk_size):
                    idx_c = valid_indices[start : start + chunk_size]
                    g_c = g_in_flat[idx_c]

                    if not g_c.any():
                        continue

                    h_c = hidden_flat[idx_c]
                    tgt_c = targets_flat[idx_c]

                    logits_c = (h_c @ weight.t()).float()
                    if bias is not None:
                        logits_c = logits_c + bias.float()

                    # d(log p_target) / dz = target_one_hot - probs = -probs + target_one_hot
                    # 原地取负后通过 scatter_add_ 在目标位置加上 1.0，彻底消除数百 MB 的 Dense One-Hot 张量
                    d_logits = F.softmax(logits_c, dim=-1).neg_()
                    d_logits.scatter_add_(
                        dim=1,
                        index=tgt_c.unsqueeze(1),
                        src=torch.ones((tgt_c.shape[0], 1), dtype=d_logits.dtype, device=d_logits.device),
                    )
                    d_logits.mul_(g_c.unsqueeze(-1))
                    d_logits_compute = d_logits.to(h_c.dtype)

                    if needs_h_grad:
                        grad_hidden_flat[idx_c] = d_logits_compute @ weight
                    if needs_w_grad:
                        grad_weight.add_(d_logits.t() @ h_c.float())
                    if needs_b_grad:
                        grad_bias.add_(d_logits.sum(dim=0))

        return (
            grad_hidden,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if grad_bias is not None else None,
            None, None, None
        )


def chunked_log_probs(
        hidden_states: torch.Tensor,
        lm_head_weight: torch.Tensor,
        targets: torch.Tensor,
        lm_head_bias: Optional[torch.Tensor] = None,
        chunk_size: int = 256,
        ignore_index: int = -100
) -> torch.Tensor:
    """
    分块计算目标 token 的 log_probs：
    hidden_states: [B, S, H] 或 [N, H]
    lm_head_weight: [V, H]
    targets: [B, S] 或 [N]
    Returns:
        log_probs: 形状与 targets 一致 ([B, S] 或 [N])
    """
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
            ignore_index
        )

    return flat_log_probs.view(orig_shape)


class ChunkedLMLoss(nn.Module):
    """
    支持 Active Token Filtering + Chunked Cross Entropy + Chunked Knowledge Distillation
    采用自定义 ChunkedCrossEntropyFunction (torch.autograd.Function) 实现:
    1. 前向仅一次 maybe_gather_lm_head_ctx，分块 GEMM，累加标量损失，不保存中间大 Logits。
    2. 反向显式进入一次 maybe_gather_lm_head_ctx，安全进行分块重算与梯度累积。
    3. 彻底规避 torch.utils.checkpoint 与 ZeRO-3 状态机的死锁和内存提前释放缺陷。
    """

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
        """
        Args:
            hidden_states: [B, S, H] 或 [N, H]
            lm_head_weight: [V, H]
            labels: [B, S] 或 [N]
            lm_head_bias: [V] (可选)
            teacher_logits: [B, S, V] 或 [N, V] (可选)
            kd_coef: 知识蒸馏权重 (0.0 ~ 1.0)
        Returns:
            total_loss: 结合 CE 和 KD 的总损失 (标量)
            ce_loss: 纯 CE 损失 (标量)
        """
        if hidden_states.dim() == 3:
            hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            if teacher_logits is not None and kd_coef > 0.0:
                shift_teacher_logits = teacher_logits[:, :-1, :].reshape(-1, teacher_logits.size(-1))
            else:
                shift_teacher_logits = None
        else:
            hidden = hidden_states
            shift_labels = labels
            shift_teacher_logits = teacher_logits

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
            # ORPO: Odds Ratio Preference Optimization
            # Odds = P / (1 - P) => log(Odds) = log(P) - log(1 - P) = p - log1mexp(p)
            p_chosen = torch.clamp(policy_chosen_means.float(), max=-1e-7)
            log_odds_chosen = p_chosen - log1mexp(p_chosen)

            p_reject = torch.clamp(policy_reject_means.float(), max=-1e-7)
            log_odds_rejected = p_reject - log1mexp(p_reject)

            log_odds_ratio = log_odds_chosen - log_odds_rejected
            losses = self.beta * -F.logsigmoid(log_odds_ratio)
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
        if loss_type == "cispo":
            # ScaleRL paper (https://arxiv.org/abs/2510.13786) recommend epsilon_max = 5.0 for CISPO.
            # If clip_eps_high is not specified or keeps the default <= 1.0 (e.g. 0.28 from DAPO/GRPO configs),
            # adaptively initialize to the recommended 5.0.
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
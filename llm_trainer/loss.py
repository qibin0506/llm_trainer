from typing import Optional, Tuple
import math

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .partition_utils import maybe_gather_lm_head_ctx


def _chunk_ce_kd_forward(
        h_chunk: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        lbl_chunk: torch.Tensor,
        t_logits_chunk: Optional[torch.Tensor],
        ignore_index: int,
        kd_coef: float
):
    """
    单个 chunk 内部的前向计算：运行在 torch.utils.checkpoint 内部
    - 投影 GEMM 保持在 Tensor Core 原始精度 (bf16/fp16)，计算后转 float32 保证损失数值稳定性
    - 同时完成 Student 投影、CE 损失与安全 KD 蒸馏损失计算
    """
    with maybe_gather_lm_head_ctx(w, b):
        s_logits = (h_chunk @ w.t()).float()
        if b is not None:
            s_logits = s_logits + b.float()

    s_log_p = F.log_softmax(s_logits, dim=-1)
    chunk_ce_loss = F.nll_loss(s_log_p, lbl_chunk, ignore_index=ignore_index, reduction="sum")

    if t_logits_chunk is not None and kd_coef > 0.0:
        valid = (lbl_chunk != ignore_index).float()
        t_probs = F.softmax(t_logits_chunk.float(), dim=-1)
        safe_s_log_p = torch.clamp(s_log_p, min=-100.0)
        safe_prod = torch.where(t_probs > 0, t_probs * safe_s_log_p, torch.zeros_like(safe_s_log_p))

        per_token_kd = -torch.sum(safe_prod, dim=-1)
        per_token_kd = torch.where(valid > 0, per_token_kd, torch.zeros_like(per_token_kd))
        chunk_kd_loss = per_token_kd.sum()

        chunk_total_loss = (1.0 - kd_coef) * chunk_ce_loss + kd_coef * chunk_kd_loss
    else:
        chunk_total_loss = chunk_ce_loss

    return chunk_total_loss, chunk_ce_loss


class ChunkedLMLoss(nn.Module):
    """
    支持 Active Token Filtering + Chunked Cross Entropy + Chunked Knowledge Distillation
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
            hidden_states: [B, S, H]
            lm_head_weight: [V, H]
            labels: [B, S]
            lm_head_bias: [V] (可选)
            teacher_logits: [B, S, V] (可选，Teacher 模型生成的软标签)
            kd_coef: 知识蒸馏权重 (0.0 ~ 1.0)
        Returns:
            total_loss: 结合 CE 和 KD 的总损失 (标量)
            ce_loss: 纯 CE 损失 (标量)
        """
        hidden = hidden_states[:, :-1, :].reshape(-1, hidden_states.size(-1))
        shift_labels = labels[:, 1:].reshape(-1)
        if teacher_logits is not None and kd_coef > 0.0:
            shift_teacher_logits = teacher_logits[:, :-1, :].reshape(-1, teacher_logits.size(-1))
        else:
            shift_teacher_logits = None

        valid = shift_labels != self.ignore_index
        n_valid_tensor = valid.sum()
        order = valid.to(torch.int8).argsort(descending=True, stable=True)
        hidden = hidden[order]
        shift_labels = shift_labels[order]
        if shift_teacher_logits is not None:
            shift_teacher_logits = shift_teacher_logits[order]

        n_valid_chunks = (n_valid_tensor + self.chunk_size - 1) // self.chunk_size

        if dist.is_available() and dist.is_initialized():
            max_chunks_tensor = n_valid_chunks.clone()
            dist.all_reduce(max_chunks_tensor, op=dist.ReduceOp.MAX)
            max_num_chunks = int(max_chunks_tensor.item())
        else:
            max_num_chunks = int(n_valid_chunks.item())

        if max_num_chunks == 0:
            with maybe_gather_lm_head_ctx(lm_head_weight, lm_head_bias):
                loss = (hidden_states.float().sum() + lm_head_weight.float().sum()) * 0.0
                if lm_head_bias is not None:
                    loss = loss + lm_head_bias.float().sum() * 0.0
            return loss, loss

        max_padded = max_num_chunks * self.chunk_size
        if max_padded > hidden.size(0):
            pad_len = max_padded - hidden.size(0)
            hidden = torch.cat([hidden, hidden.new_zeros((pad_len, hidden.size(-1)))], dim=0)
            shift_labels = torch.cat([shift_labels, shift_labels.new_full((pad_len,), self.ignore_index)], dim=0)
            if shift_teacher_logits is not None:
                shift_teacher_logits = torch.cat([
                    shift_teacher_logits,
                    shift_teacher_logits.new_zeros((pad_len, shift_teacher_logits.size(-1)))
                ], dim=0)
        elif max_padded < hidden.size(0):
            hidden = hidden[:max_padded]
            shift_labels = shift_labels[:max_padded]
            if shift_teacher_logits is not None:
                shift_teacher_logits = shift_teacher_logits[:max_padded]

        total_loss_accum = hidden.new_zeros((), dtype=torch.float32)
        ce_loss_accum = hidden.new_zeros((), dtype=torch.float32)

        for start in range(0, max_padded, self.chunk_size):
            h_chunk = hidden[start: start + self.chunk_size]
            lbl_chunk = shift_labels[start: start + self.chunk_size]
            t_chunk = shift_teacher_logits[start: start + self.chunk_size] if shift_teacher_logits is not None else None
            chunk_total, chunk_ce = torch_checkpoint(
                _chunk_ce_kd_forward,
                h_chunk,
                lm_head_weight,
                lm_head_bias,
                lbl_chunk,
                t_chunk,
                self.ignore_index,
                kd_coef,
                use_reentrant=False,
            )
            total_loss_accum = total_loss_accum + chunk_total
            ce_loss_accum = ce_loss_accum + chunk_ce

        if n_valid_tensor > 0:
            total_loss = total_loss_accum / n_valid_tensor
            ce_loss = ce_loss_accum / n_valid_tensor
        else:
            total_loss = total_loss_accum * 0.0
            ce_loss = ce_loss_accum * 0.0
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
            ref_chosen_logps: torch.Tensor,
            ref_reject_logps: torch.Tensor,
            policy_chosen_means: Optional[torch.Tensor] = None,
            policy_reject_means: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.loss_type == 'orpo':
            # ORPO: Odds Ratio Preference Optimization
            # Odds = P / (1 - P) => log(Odds) = log(P) - log(1 - P)
            p_chosen = torch.clamp(policy_chosen_means.float(), max=-1e-6)
            log_odds_chosen = p_chosen - torch.log1p(-torch.exp(p_chosen))

            p_reject = torch.clamp(policy_reject_means.float(), max=-1e-6)
            log_odds_rejected = p_reject - torch.log1p(-torch.exp(p_reject))

            log_odds_ratio = log_odds_chosen - log_odds_rejected
            losses = self.beta * -F.logsigmoid(log_odds_ratio)
            return losses.mean()

        if self.loss_type == 'simpo':
            # SimPO: Simple Preference Optimization
            reward_diff = self.beta * (policy_chosen_means.float() - policy_reject_means.float())
            losses = -F.logsigmoid(reward_diff - self.simpo_gamma)
            return losses.mean()

        policy_chosen_logps = policy_chosen_logps.float()
        policy_reject_logps = policy_reject_logps.float()
        ref_chosen_logps = ref_chosen_logps.float()
        ref_reject_logps = ref_reject_logps.float()

        pi_logratios = policy_chosen_logps - policy_reject_logps
        ref_logratios = ref_chosen_logps - ref_reject_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
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
            huber_delta: float = 1.0
    ):
        """
        初始化PPO损失函数。
        :param clip_eps: PPO裁剪范围的epsilon值。
        :param vf_coef: 价值函数损失的系数。
        """
        super().__init__()
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.huber_delta = huber_delta

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
        :return: (总损失, Actor损失, Value损失, Entropy)
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

        # Value Loss (价值损失) with clipping
        values_clipped = old_values + torch.clamp(values - old_values, -self.clip_eps, self.clip_eps)

        vf_loss_unclipped = F.smooth_l1_loss(values, returns, reduction='none', beta=self.huber_delta)
        vf_loss_clipped = F.smooth_l1_loss(values_clipped, returns, reduction='none', beta=self.huber_delta)
        value_loss = torch.max(vf_loss_unclipped, vf_loss_clipped)

        # Apply mask and average
        value_loss = 0.5 * (value_loss * value_mask).sum() / value_mask.sum().clamp(min=1.0)
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

            # 计算裁剪比例
            clipped = ratio.gt(1.0 + self.clip_eps) | ratio.lt(1.0 - self.clip_eps)
            clip_frac = torch.sum(clipped.float() * mask) / mask.sum().clamp(min=1.0)

            entropy = -torch.sum(log_probs * mask) / mask.sum().clamp(min=1.0)

        return total_loss, actor_loss, value_loss, approx_kl, clip_frac, entropy


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

        is_nonneg_adv = advantages >= 0
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
            completion_len: int
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

        coef_1 = torch.exp(log_importance_weights)

        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=1 + self.clip_eps_high).detach()
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
            coef_2 = torch.clamp(coef_1, 1 - self.clip_eps_low, 1 + self.clip_eps_high)
            if self.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.delta)

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
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            completion_len = max(completion_len, 1)
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * completion_len)
        elif self.loss_type == "luspo":
            loss = (per_token_loss * completion_mask).sum(dim=-1).mean()
        else:
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        with torch.no_grad():
            is_clipped = (coef_1 > 1 + self.clip_eps_high) | (coef_1 < 1 - self.clip_eps_low)
            clip_frac = (is_clipped.float() * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)

        return loss, clip_frac
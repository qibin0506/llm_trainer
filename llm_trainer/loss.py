from typing import List, Optional
import math
import torch
from torch import nn
import torch.nn.functional as F


class LMLoss(nn.Module):
    """
    llm loss
    """
    def __init__(
            self,
            ignore_index: int = -100,
            *,
            critical_tokens: Optional[List[int]] = None,
            critical_alpha: float = 1.0,
            vocab_size: int = 0
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.critical_tokens = critical_tokens
        self.critical_alpha = critical_alpha

        if critical_tokens and vocab_size > 0:
            self.register_buffer('weights', torch.ones(vocab_size))
            # 为关键token设置权重
            self.weights[self.critical_tokens] = critical_alpha


    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits shape (batch, seq_len, vocab_size)
        # labels shape (batch, seq_len)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits = shift_logits.reshape(-1, logits.shape[-1])
        targets = shift_labels.reshape(-1)

        ce_loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index,
            weight=self.weights.to(logits.device, dtype=logits.dtype) if self.critical_tokens else None
        )

        # 添加额外惩罚项（可选）
        # if self.critical_tokens:
        #     crit_mask = torch.isin(targets, torch.tensor(self.critical_tokens).to(targets.device))
        #     crit_logits = logits[crit_mask]
        #     crit_targets = targets[crit_mask]
        #     extra_loss = F.cross_entropy(crit_logits, crit_targets, ignore_index=self.ignore_index)
        #     return ce_loss + extra_loss * (self.critical_alpha - 1)  # 增强惩罚

        return ce_loss


class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/models/loss.py#L266
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)

        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)

        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (labels != self.ignore_index).float().view(-1)
        distil_loss = -torch.sum(x * mask) / mask.sum().clamp(min=1e-8)

        return distil_loss


class DPOLoss(nn.Module):
    def __init__(
            self,
            beta: float,
            label_smoothing: float = 0.0,
            ipo: bool = False
    ):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
            self,
            policy_chosen_logps: torch.Tensor,
            policy_reject_logps: torch.Tensor,
            ref_chosen_logps: torch.Tensor,
            ref_reject_logps: torch.Tensor
    ) -> torch.Tensor:
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
    ):
        """
        初始化PPO损失函数。
        :param clip_eps: PPO裁剪范围的epsilon值。
        :param vf_coef: 价值函数损失的系数。
        """
        super().__init__()
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef

    def forward(
            self,
            log_probs: torch.Tensor,
            old_log_probs: torch.Tensor,
            values: torch.Tensor,
            old_values: torch.Tensor,
            returns: torch.Tensor,
            advantages: torch.Tensor,
            mask: torch.Tensor
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
        log_probs = log_probs.float()
        old_log_probs = old_log_probs.float()
        values = values.float()
        old_values = old_values.float()
        returns = returns.float()
        advantages = advantages.float()
        mask = mask.float()

        # Value Loss (价值损失) with clipping
        values_clipped = old_values + torch.clamp(values - old_values, -self.clip_eps, self.clip_eps)
        vf_loss_unclipped = F.mse_loss(values, returns, reduction='none')
        vf_loss_clipped = F.mse_loss(values_clipped, returns, reduction='none')
        value_loss = torch.max(vf_loss_unclipped, vf_loss_clipped)
        # Apply mask and average
        value_loss = 0.5 * (value_loss * mask).sum() / mask.sum().clamp(min=1.0)
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
        self.clip_eps_high = clip_eps_high if clip_eps_high else clip_eps_low
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
    ) -> torch.Tensor:
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
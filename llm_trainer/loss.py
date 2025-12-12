from typing import List, Optional
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
        mask = (labels != self.ignore_index).int()

        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

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

        return total_loss, actor_loss, value_loss, approx_kl, clip_frac


class GRPOLoss(nn.Module):
    def __init__(
            self,
            beta: float,
            clip_eps_low: float,
            clip_eps_high: Optional[float] = None,
            delta: Optional[float] = None,
            importance_sampling_level: str = 'token',
            loss_type: str = 'grpo',
            gen_max_new_tokens: Optional[float] = None
    ):
        super().__init__()

        self.beta = beta
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high if clip_eps_high else clip_eps_low
        self.delta = delta
        self.importance_sampling_level = importance_sampling_level
        self.loss_type = loss_type
        self.gen_max_new_tokens = gen_max_new_tokens

    def forward(
            self,
            log_probs: torch.Tensor,
            old_log_probs: torch.Tensor,
            ref_log_probs: torch.Tensor,
            completion_mask: torch.Tensor,
            advantages: torch.Tensor
    ) -> torch.Tensor:

        if self.beta != 0.0:
            per_token_kl = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        else:
            per_token_kl = None

        log_ratio = log_probs - old_log_probs
        if self.importance_sampling_level == "seq":
            # GSPO
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            # GRPO
            log_importance_weights = log_ratio

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.clip_eps_low, 1 + self.clip_eps_high)

        # Two-sided clipping
        if self.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.delta)

        per_token_loss1 = coef_1 * advantages
        per_token_loss2 = coef_2 * advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            assert self.gen_max_new_tokens is not None
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.gen_max_new_tokens)
        else:
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()

        return loss

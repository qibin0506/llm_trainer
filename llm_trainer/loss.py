from typing import List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from llama import LlamaModel


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

    def _log_probs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
        if logits.dtype in [torch.float32, torch.float64]:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.stack(
                [torch.logsumexp(l, dim=-1) for l in logits]  # loop to reduce peak mem consumption
            )
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            log_probs_labels = []
            for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
                row_log_probs = F.log_softmax(row_logits, dim=-1)
                row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                log_probs_labels.append(row_log_probs_labels)
            log_probs_labels = torch.stack(log_probs_labels)
        return log_probs_labels

    def logprobs(self, logits, labels, mask):
        """
        Calculate the average log probabilities for a batch of sequences.

        Args:
            logits (torch.Tensor): Logits from the model with shape (B, T, V)
            labels (torch.Tensor): Ground truth labels with shape (B, T).
            mask (torch.Tensor): Mask tensor with shape (B, T) indicating
                which tokens are not padding (1 for valid tokens, 0 for padding).

        Returns:
            torch.Tensor: Average log probabilities for each sequence in the batch.
                          Shape is (B,) representing the mean log probability for each sequence.
        """
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        # Shift mask right by one to align with labels
        mask = mask[:, 1:].clone()

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        # Gather the log probabilities for the actual labels
        per_token_logps = self._log_probs_from_logits(logits, labels)

        # Apply the mask to set log-probs of padding tokens to 0
        logprobs_sums = (per_token_logps * mask).sum(-1)

        logprobs_means = (per_token_logps * mask).sum(-1) / mask.sum(-1)

        return logprobs_sums, -logprobs_means.mean()

    def forward(
            self,
            policy_chosen_logps: torch.Tensor,
            policy_rejected_logps: torch.Tensor,
            reference_chosen_logps: torch.Tensor,
            reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
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
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


class GRPOLoss(nn.Module):
    def __init__(
            self,
            clip_eps: float,
            kl_weight: float
    ):
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(
            self,
            log_probs: torch.Tensor,
            old_log_probs: torch.Tensor,
            ref_log_probs: torch.Tensor,
            completion_mask: torch.Tensor,
            advantages: torch.Tensor
    ) -> torch.Tensor:
        # Compute policy ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute surrogate loss with clipping
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        surrogate_loss = torch.min(surrogate1, surrogate2)

        # Compute KL divergence penalty
        kl_div = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1

        # Combine losses
        per_token_loss = surrogate_loss - self.kl_weight * kl_div
        loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        return loss

        # kl = self._approx_kl_divergence(
        #     log_probs=log_probs,
        #     ref_log_probs=ref_log_probs,
        #     mask=mask,
        # )
        #
        # ratio = (log_probs - old_log_probs).exp()
        # surr1 = ratio * advantages
        # surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        # loss = -torch.min(surr1, surr2) + self.kl_weight * kl
        #
        # loss = self._masked_mean(loss, mask, dim=-1).mean()
        # return loss, kl.mean()

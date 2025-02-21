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

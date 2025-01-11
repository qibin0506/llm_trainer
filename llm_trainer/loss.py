from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

# def pretrain_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#     logits = logits.reshape(-1, logits.shape[-1])
#     targets = labels.reshape(-1)
#
#     return F.cross_entropy(logits, targets, ignore_index=-100)


class LMLoss(nn.Module):
    """
    llm loss
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits shape (batch, seq_len, vocab_size)
        # labels shape (batch, seq_len)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        logits = shift_logits.reshape(-1, logits.shape[-1])
        targets = shift_labels.reshape(-1)

        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index)


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

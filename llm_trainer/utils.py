import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .tokenizer import Tokenizer
from .parallel import Parallel
from .parallel_fsdp import FsdpParallel
from .parallel_ddp import DdpParallel
from .parallel_none import NoneParallel


class TrainerTools:
    def __init__(self):
        if not hasattr(TrainerTools, "_first_init"):
            TrainerTools._first_init = True

            parallel_type = os.environ.get('PARALLEL_TYPE', 'none')
            if parallel_type == 'fsdp':
                self.parallel: Parallel = FsdpParallel()
            elif parallel_type == 'ddp':
                self.parallel: Parallel = DdpParallel()
            else:
                self.parallel: Parallel = NoneParallel()

            self.tokenizer = Tokenizer(int(os.environ.get('TOKENIZERS_TYPE', 0)))

            self.use_amp = 'cuda' in self.parallel.device

            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            self.bot_token = self.tokenizer.encode_to_token('[BOT]', unsqueeze=False, covert_tensor=False)[0]


    def __new__(cls, *args, **kwargs):
        if not hasattr(TrainerTools, "_instance"):
            TrainerTools._instance = object.__new__(cls)

        return TrainerTools._instance

def pretrain_padding_fn(batch_data):
    inputs = pad_sequence(batch_data, batch_first=True, padding_value=TrainerTools().tokenizer.pad)
    # crossEntropy默认的ignore_index是-100
    labels = pad_sequence(batch_data, batch_first=True, padding_value=-100)

    return inputs, labels


def sft_padding_fn(batch_data):
    """
     如果是sft，则不计算prompt部分的loss, 例如：
    logits: [USER]:你好[BOT]:我好[SEP]
    labels: [USER]:你好[BOT]:我好[SEP]

    shift_logits: [USER]:你好[BOT]:我好
    shift_labels: :你好[BOT]:我好[SEP]

    mask_labels: mask mask mask mask:我好[SEP]
        * [BOT]后的:暂时不考虑，后续可以把prompt里的：去掉
        * mask=-100和pad一样
    """
    inputs, labels = pretrain_padding_fn(batch_data)
    batch_size = len(labels)
    batch_bot_idx = torch.nonzero(torch.eq(labels, TrainerTools().bot_token), as_tuple=True)[1]

    for batch in range(batch_size):
        bot_idx = batch_bot_idx[batch].item() + 1
        labels[batch, :bot_idx] = torch.tensor([-100] * bot_idx)

    return inputs, labels


# def pretrain_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#     logits = logits.reshape(-1, logits.shape[-1])
#     targets = labels.reshape(-1)
#
#     return F.cross_entropy(logits, targets, ignore_index=-100)


def calc_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits shape (batch, seq_len, vocab_size)
    # labels shape (batch, seq_len)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    logits = shift_logits.reshape(-1, logits.shape[-1])
    targets = shift_labels.reshape(-1)

    return F.cross_entropy(logits, targets, ignore_index=-100)


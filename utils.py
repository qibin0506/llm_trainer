import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from pytorch.llm.llm_trainer.tokenizer import Tokenizer
from pytorch.llm.llm_trainer.ddp import DDPHelper


class TrainConfig:
    def __init__(self):
        if not hasattr(TrainConfig, "_first_init"):
            TrainConfig._first_init = True

            # 梯度积累步数
            self.gradient_accumulation_steps = self._get_env_int('GRADIENT_ACCUMULATION_STEPS', 0)

            self.ddp_helper = DDPHelper()
            self.tokenizer = Tokenizer(self._get_env_int('TOKENIZERS_TYPE', 0))

            self.use_amp = 'cuda' in self.ddp_helper.device
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            self.bot_token = self.tokenizer.encode_to_token('[BOT]', unsqueeze=False, covert_tensor=False)[0]

    def _get_env_int(self, name, default_value: int) -> int:
        try:
            value = os.environ[name]
            return int(value)
        except:
            return default_value

    def __new__(cls, *args, **kwargs):
        if not hasattr(TrainConfig, "_instance"):
            TrainConfig._instance = object.__new__(cls)

        return TrainConfig._instance

def pretrain_padding_fn(batch_data):
    inputs = pad_sequence(batch_data, batch_first=True, padding_value=TrainConfig().tokenizer.pad)
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
    batch_bot_idx = torch.nonzero(torch.eq(labels, TrainConfig().bot_token), as_tuple=True)[1]

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


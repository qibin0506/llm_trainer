import torch
from torch.nn.utils.rnn import pad_sequence
from .tools import TrainerTools
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pretrain_collate_fn(batch_data):
    inputs = pad_sequence(batch_data, batch_first=True, padding_value=TrainerTools().tokenizer.pad)
    # crossEntropy默认的ignore_index是-100
    labels = pad_sequence(batch_data, batch_first=True, padding_value=-100)

    return inputs, labels


def sft_collate_fn(batch_data):
    """
     如果是sft，则不计算prompt部分的loss, 例如：
    logits: [USER]你好[BOT]我好[SEP]
    labels: [USER]你好[BOT]我好[SEP]

    shift_logits: [USER]你好[BOT]我好
    shift_labels: 你好[BOT]我好[SEP]

    mask_labels: mask mask mask mask 我好[SEP]
        * mask=-100和pad一样


    多轮对话场景
    [USER]你好[BOT]我好[SEP][USER]很好[BOT]不好[SEP]
    mask: mask mask mask mask 我好[SEP] mask mask mask mask 不好[SEP]
    """

    inputs, labels = pretrain_collate_fn(batch_data)

    # 支持多轮会话的mask
    for batch, label in enumerate(labels):
        start_index = -1
        for index, token in enumerate(label):
            if token == TrainerTools().tokenizer.user:
                start_index = index
            elif token == TrainerTools().tokenizer.bot and start_index != -1:
                labels[batch, start_index:index + 1] = -100
                start_index = -1

    # 支持单轮会话的mask
    # inputs, labels = pretrain_collate_fn(batch_data)
    # batch_size = len(labels)
    # batch_bot_idx = torch.nonzero(torch.eq(labels, TrainerTools().tokenizer.bot), as_tuple=True)[1]
    #
    # for batch in range(batch_size):
    #     bot_idx = batch_bot_idx[batch].item() + 1
    #     labels[batch, :bot_idx] = torch.tensor([-100] * bot_idx)

    return inputs, labels

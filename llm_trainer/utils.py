import time
import torch
from torch.nn.utils.rnn import pad_sequence
from .train_tools import TrainerTools
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    # 如果使用多 GPU
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


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


def log(msg: str, log_file=None):
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    if log_file is None:
        print(f'({cur_time}) msg')
    else:
        with open(log_file, 'a') as f:
            f.write(f"({cur_time}) {msg}")

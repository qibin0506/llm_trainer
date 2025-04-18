import os
import random
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from .tools import TrainerTools
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pretrain_collate_fn(batch_data):
    # [[x,x,x], [y,y,y]]
    inputs = pad_sequence(batch_data, batch_first=True, padding_value=TrainerTools().tokenizer.pad)
    # crossEntropy默认的ignore_index是-100
    labels = pad_sequence(batch_data, batch_first=True, padding_value=-100)

    return inputs, labels


def _mask_prompt(labels):
    tokenizer = TrainerTools().tokenizer
    # 支持多轮会话的mask
    for batch, label in enumerate(labels):
        start_index = -1
        for index, token in enumerate(label):
            if token == tokenizer.system or token == tokenizer.user:
                start_index = index
            elif token == tokenizer.end and start_index != -1:
                labels[batch, start_index:index + 1] = -100
                start_index = -1

    return labels


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
    labels = _mask_prompt(labels)

    # 支持单轮会话的mask
    # inputs, labels = pretrain_collate_fn(batch_data)
    # batch_size = len(labels)
    # batch_bot_idx = torch.nonzero(torch.eq(labels, TrainerTools().tokenizer.assistant), as_tuple=True)[1]
    #
    # for batch in range(batch_size):
    #     bot_idx = batch_bot_idx[batch].item() + 1
    #     labels[batch, :bot_idx] = torch.tensor([-100] * bot_idx)

    return inputs, labels


def dpo_collate_fn(batch_data):
    # batch_data: [{'chosen': chosen, 'rejected': rejected}, {'chosen': chosen, 'rejected': rejected}]
    chosen_inputs = []
    chosen_labels = []
    rejected_inputs = []
    rejected_labels = []

    max_len = 0
    for key in ['chosen', 'rejected']:
        max_len = max(max(len(item[key]) for item in batch_data), max_len)

    for item in batch_data:
        chosen_sequence = item['chosen']
        chosen_inputs.append(chosen_sequence + [TrainerTools().tokenizer.pad] * (max_len - len(chosen_sequence)))
        chosen_labels.append(chosen_sequence + [-100] * (max_len - len(chosen_sequence)))

        rejected_sequence = item['rejected']
        rejected_inputs.append(rejected_sequence + [TrainerTools().tokenizer.pad] * (max_len - len(rejected_sequence)))
        rejected_labels.append(rejected_sequence + [-100] * (max_len - len(rejected_sequence)))

    chosen_inputs = torch.tensor(chosen_inputs).long()
    chosen_labels = _mask_prompt(torch.tensor(chosen_labels).long())

    rejected_inputs = torch.tensor(rejected_inputs).long()
    rejected_labels = _mask_prompt(torch.tensor(rejected_labels).long())

    return {
        'chosen_inputs': chosen_inputs,
        'chosen_labels': chosen_labels,
        'rejected_inputs': rejected_inputs,
        'rejected_labels': rejected_labels
    }

def _zero_pad_sequences(
    sequences: list[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def split_batch(data_per_batch: dict) -> list[dict]:
    """
    from: data_per_batch("sequences": [group_size, max_generate_len] ...)
    to:   [dict("sequences": [max_generate_len] ...) ... group_size]
    """

    group_size = data_per_batch['sequence_ids'].size(0)
    # [{"sequence_ids": xxx, "old_log_probs": xxx...}, ...]
    group_data = [{} for _ in range(group_size)]

    keys = (
        'sequence_ids',
        'old_log_probs',
        'ref_log_probs',
        'advantages',
        'attention_mask',
        'mask',
    )

    for key in keys:
        value = data_per_batch[key]
        if value is None:
            vals = [None] * group_size
        else:
            vals = torch.unbind(value)

        for i, v in enumerate(vals):
            group_data[i][key] = v

    return group_data


def join_batch(batch_data: list[dict]) -> dict:
    """
    from: [dict("sequences": [max_generate_len] ...), ...]
    to:   dict("sequences": max_generate_len, ...)
    """

    result = {}
    keys = (
        'sequence_ids',
        'old_log_probs',
        'ref_log_probs',
        'advantages',
        'attention_mask',
        'mask',
    )

    for key in keys:
        # [sequence_ids, sequence_ids ...]
        # shape [batch_size, seq_len]
        vals = [item[key] for item in batch_data]
        if all(v is not None for v in vals):
            data = _zero_pad_sequences(vals, "left")
        else:
            data = None
        result[key] = data

    return result

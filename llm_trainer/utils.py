import random
from contextlib import nullcontext
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


def autocast(device_type):
    if TrainerTools().use_amp:
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=True,
            cache_enabled=None
        )
    else:
        return nullcontext()


def create_doc_boundary_mask(
        input_ids: torch.Tensor,
        dtype: torch.dtype
) -> torch.Tensor:
    """
    根据文档结束符 (eot) 的位置，创建一个 attention mask 来阻止跨文档的注意力。

    这个函数生成的 mask 会阻止一个 token 关注 (attend to) 属于前面文档的 tokens。
    例如，对于输入 `[[1, 2, eot, 3, 4, eot]]`，
    tokens `3` 和 `4` 将无法关注 `1`, `2`, 和第一个 `eot`。

    Args:
        input_ids (torch.Tensor): 输入的 token ID 张量，形状为 (bsz, seq_len)。
        dtype (torch.dtype): 数据类型。

    Returns:
        torch.Tensor: 符合 attention 机制要求的 mask 张量，
                      形状为 (bsz, 1, seq_len, seq_len)。
                      值为 -inf 的位置表示被屏蔽，值为 0 的位置表示允许注意力。
    """
    # 获取 batch size 和 sequence length
    bsz, seq_len = input_ids.shape

    # 1. 确定每个 eot_token 的位置
    # is_eot 是一个布尔张量，形状为 (bsz, seq_len)
    is_eot = (input_ids == TrainerTools().tokenizer.end)

    # 2. 为每个 token 分配一个文档 ID
    # 我们使用 cumsum (累加和) 来创建递增的文档 ID。一个 token 所属的文档 ID，
    # 取决于它前面有多少个 eot。
    # 示例:
    # input_ids:        [[1, 2, 3, eot, 4, 5, eot]]
    # is_eot:           [F, F, F, T, F, F, T] -> [0, 0, 0, 1, 0, 0, 1]
    # doc_ids_ending:   [0, 0, 0, 1, 1, 1, 2] (cumsum 的结果)
    # doc_ids:          [0, 0, 0, 0, 1, 1, 1] (向右移位后的结果)
    # 这个结果正确地将文档 0 分配给了前四个 token，将文档 1 分配给了后三个 token。
    doc_ids_ending = torch.cumsum(is_eot, dim=-1)
    doc_ids = F.pad(doc_ids_ending[:, :-1], (1, 0), value=0)

    # 3. 通过比较 query 和 key 的文档 ID 来创建 mask
    # 我们的目标是：当 query token 所在的文档 ID 大于 key token 所在的文档 ID 时，进行屏蔽。
    # query_doc_ids 形状: (bsz, seq_len, 1)
    # key_doc_ids 形状:   (bsz, 1, seq_len)
    query_doc_ids = doc_ids.unsqueeze(2)
    key_doc_ids = doc_ids.unsqueeze(1)

    # 利用 PyTorch 的广播机制，`query_doc_ids > key_doc_ids` 会创建一个
    # 形状为 (bsz, seq_len, seq_len) 的布尔张量。
    # 当 query 的文档 ID 大于 key 的文档 ID 时，值为 True，这正是我们需要屏蔽的位置。
    boundary_mask = query_doc_ids > key_doc_ids

    # 4. 将布尔 mask 转换为 attention 机制所需的浮点数 mask (-inf 和 0)
    final_mask = torch.zeros(
        (bsz, seq_len, seq_len), device=input_ids.device, dtype=dtype
    )
    final_mask.masked_fill_(boundary_mask, torch.finfo(dtype).min)

    # 5. 增加一个维度以匹配 attention head 的输入要求 (bsz, num_heads, seq_len, seq_len)
    #    这里我们只生成一个 mask，它可以被广播到所有的 head。
    return final_mask.unsqueeze(1)


def generate_position_ids(input_ids: torch.Tensor):
    """
    为打包序列生成 position_ids 张量。

    参数:
      input_ids (torch.Tensor): 输入的 token ID 张量 (batch_size, sequence_length)。
      end_of_text_id (int): 代表文本结束的特殊 token ID。

    返回:
      torch.Tensor: 生成的 position_ids 张量。
    """
    # 获取输入张量的形状
    batch_size, seq_length = input_ids.shape

    # 创建一个与输入形状相同，全为0的张量来存储position_ids
    # 第一个token的位置永远是0，所以这个初始化是正确的
    position_ids = torch.zeros_like(input_ids, dtype=torch.long)

    # 从第二个时间步 (t=1) 开始遍历整个序列
    for t in range(1, seq_length):
        # 检查前一个时间步 (t-1) 的token是否为 EOT token
        # 这会为批次中的每个序列生成一个布尔值
        is_reset_token = (input_ids[:, t - 1] == TrainerTools().tokenizer.end)

        # 获取前一个时间步的位置ID
        prev_position_ids = position_ids[:, t - 1]

        # 如果前一个token是EOT，当前位置重置为0；否则，在前一个位置上加1
        # torch.where 会根据 is_reset_token 的布尔值进行选择
        position_ids[:, t] = torch.where(is_reset_token, 0, prev_position_ids + 1)

    return position_ids


def repeat_image_tok(
        tokens: torch.Tensor,
        tokens_per_image: int
) -> torch.Tensor:
    # tokens_per_image=3 -> <image>...xxxx -> <image><image><image>...xxx
    image_tok = TrainerTools().tokenizer.image
    if image_tok not in tokens:
        return tokens

    image_tok_idx = torch.where(tokens == image_tok)[0].item()
    repeat_image_toks = torch.tensor([image_tok] * tokens_per_image, dtype=tokens.dtype, device=tokens.device)

    # repeat image_tok
    new_tokens = torch.concat([tokens[:image_tok_idx], repeat_image_toks, tokens[image_tok_idx + 1:]], dim=-1)
    return new_tokens


def batch_repeat_image_tok(
        tokens: torch.Tensor,
        tokens_per_image: int
) -> torch.Tensor:
    new_tokens = []

    for token in tokens:
        new_tokens.append(repeat_image_tok(token, tokens_per_image))

    return torch.stack(new_tokens, dim=0)


def pretrain_collate_fn(batch_data):
    # [[x,x,x], [y,y,y]]
    inputs = pad_sequence(batch_data, batch_first=True, padding_value=TrainerTools().tokenizer.pad)
    # crossEntropy默认的ignore_index是-100
    labels = pad_sequence(batch_data, batch_first=True, padding_value=-100)

    # inputs, labels
    return {
        'inputs': inputs,
        'labels': labels
    }


def get_sft_collate_fn(mask_prompt: bool):
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
        batch_train_data = []
        image_tags = []
        for item in batch_data:
            batch_train_data.append(item['inputs'])
            image_tags.append(item['image_tag'])

        # [[x,x,x], [y,y,y]]
        inputs = pad_sequence(batch_train_data, batch_first=True, padding_value=TrainerTools().tokenizer.pad)
        # crossEntropy默认的ignore_index是-100
        labels = pad_sequence(batch_train_data, batch_first=True, padding_value=-100)

        if mask_prompt:
            labels = _mask_prompt(labels)

        return {
            'inputs': inputs,
            'labels': labels,
            'image_tags': image_tags
        }

    return sft_collate_fn


def get_dpo_collate_fn(mask_prompt: bool):
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
        chosen_labels = torch.tensor(chosen_labels).long()
        if mask_prompt:
            chosen_labels = _mask_prompt(chosen_labels)

        rejected_inputs = torch.tensor(rejected_inputs).long()
        rejected_labels = torch.tensor(rejected_labels).long()
        if mask_prompt:
            rejected_labels = _mask_prompt(rejected_labels)

        return {
            'chosen_inputs': chosen_inputs,
            'chosen_labels': chosen_labels,
            'rejected_inputs': rejected_inputs,
            'rejected_labels': rejected_labels
        }

    return dpo_collate_fn


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


def fill_loss_mask(loss_masks, labels):
    """
    将loss_mask中prompt部分强制设置为False
    loss_masks: shape  (B, T)
    labels: shape (B, T)
    """
    tokenizer = TrainerTools().tokenizer
    # 支持多轮会话的mask
    for batch, label in enumerate(labels):
        start_index = -1
        for index, token in enumerate(label):
            if token == tokenizer.system or token == tokenizer.user:
                start_index = index
            elif token == tokenizer.end and start_index != -1:
                loss_masks[batch, start_index:index + 1] = False
                start_index = -1

    return loss_masks


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
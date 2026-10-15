import pickle
import csv
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from .tools import TrainerTools
from .utils import repeat_image_tok


"""
support jsonl and pkl
"""
def _get_file_type(file_path: str):
    if file_path.endswith('.npy'):
        return 'npy'
    elif file_path.endswith('.jsonl'):
        return 'jsonl'
    elif file_path.endswith('.pkl'):
        return 'pkl'

    return None


class PretrainDataset(Dataset):
    """
    适用于pretrain阶段，数据格式支持jsonl和pkl，如果是jsonl会在init阶段全部encode成token
    1. npy:【推荐】numpy 数组，支持 mmap，内存占用极低
    2. jsonl: {'text': 'text1'}\n{'text': 'text2'}
    3. pkl: [0, 1, 2, 3 ...]
    """
    def __init__(
            self,
            file_path,
            block_size,
            stride
    ):
        super().__init__()

        self.block_size = block_size
        self.stride = stride
        self.use_mmap = False

        file_type = _get_file_type(file_path)

        if file_type == 'npy':
            self.input_ids = np.load(file_path, mmap_mode='r')
            self.use_mmap = True
        elif file_type == 'jsonl':
            tokens = []
            with open(file_path, 'r') as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue

                    doc_tokens = TrainerTools().tokenizer.encode(json.loads(line_str)['text'])
                    if doc_tokens:
                        if doc_tokens[-1] != TrainerTools().tokenizer.end:
                            doc_tokens.append(TrainerTools().tokenizer.end)
                        tokens.extend(doc_tokens)

            self.input_ids = torch.tensor(tokens, dtype=torch.int32)
            del tokens
        elif file_type == 'pkl':
            with open(file_path, 'rb') as f:
                tokens = pickle.load(f)
            self.input_ids = torch.tensor(tokens, dtype=torch.int32)
            del tokens
        else:
            raise ValueError(f"Unsupported file type for {file_path}")

        if len(self.input_ids) < block_size:
            self.length = 0
        else:
            self.length = (len(self.input_ids) - block_size) // stride + 1

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        if item < 0 or item >= self.length:
            raise IndexError(f"Index {item} out of range")

        start_idx = item * self.stride
        end_idx = start_idx + self.block_size

        data = self.input_ids[start_idx:end_idx]

        if self.use_mmap:
            return torch.from_numpy(data.astype(np.int64))
        else:
            return data.long()


class SFTDataset(Dataset):
    """
    适用于sft阶段，数据格式支持jsonl和pkl，如果是jsonl，则会在getitem阶段encode成token
    npy: [
            [0, 1, 2, 3],
            [4, 5, 6, 7]
         ]
    jsonl: [
            {'role': 'system', 'content': 'system_content'},
            {'role': 'user', 'content': 'user_content'},
            {'role': 'assistant', 'think': 'think_content', 'content': 'assistant_content'}
           ]\n
           [
            {'role': 'system', 'content': 'system_content'},
            {'role': 'user', 'content': 'user_content'},
            {'role': 'assistant', 'think': 'think_content', 'content': 'assistant_content'}
           ]
    pkl: [
            [0, 1, 2, 3],
            [4, 5, 6, 7]
         ]
    """
    def __init__(
            self,
            file_path,
            block_size,
            image_tags_file_path=None,
            tokens_per_image=-1
    ):
        super().__init__()

        self.block_size = block_size
        self.tokens_per_image = tokens_per_image
        self.input_ids = []
        self.image_tags = []
        self.plain_text = False

        file_type = _get_file_type(file_path)

        if file_type == 'npy':
            try:
                self.input_ids = np.load(file_path, mmap_mode='r')
            except ValueError:
                self.input_ids = np.load(file_path, allow_pickle=True)
        elif file_type == 'jsonl':
            self.plain_text = True
            with open(file_path, 'r') as f:
                for line in f:
                    self.input_ids.append(json.loads(line.strip()))
        elif file_type == 'pkl':
            with open(file_path, 'rb') as f:
                self.input_ids = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type for {file_path}")

        if image_tags_file_path:
            with open(image_tags_file_path, 'r') as f:
                csv_reader = csv.reader(f)
                for line in csv_reader:
                    if line:
                        self.image_tags.append(line[0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        if self.plain_text:
            inputs = TrainerTools().tokenizer.apply_chat_template(self.input_ids[item])
        else:
            inputs = self.input_ids[item]

        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs.astype(np.int64))
        else:
            inputs = torch.tensor(inputs).long()

        image_tok = TrainerTools().tokenizer.image
        image_tag = self.image_tags[item] if (self.image_tags and item < len(self.image_tags)) else None

        if self.tokens_per_image != -1 and image_tag and (inputs == image_tok).any().item():
            inputs, _ = repeat_image_tok(inputs, self.tokens_per_image)
        else:
            image_tag = None

        if len(inputs) > self.block_size:
            inputs = inputs[:self.block_size]
            if self.tokens_per_image != -1 and image_tag:
                actual_img_tokens = (inputs == image_tok).sum().item()
                if actual_img_tokens % self.tokens_per_image != 0:
                    valid_len = actual_img_tokens - (actual_img_tokens % self.tokens_per_image)
                    if valid_len == 0:
                        inputs = inputs[inputs != image_tok]
                        image_tag = None
                    else:
                        # 移除末尾残缺的超出部分 <image> token
                        img_indices = (inputs == image_tok).nonzero(as_tuple=True)[0]
                        to_remove_indices = img_indices[valid_len:]
                        keep_mask = torch.ones_like(inputs, dtype=torch.bool)
                        keep_mask[to_remove_indices] = False
                        inputs = inputs[keep_mask]

        return {
            'inputs': inputs,
            'image_tag': image_tag
        }


class DPODataset(Dataset):
    """
    适用于dpo阶段，数据格式支持jsonl和pkl，如果是jsonl，则会在getitem阶段encode成token
    npy: [
            {'chosen': xxx, 'rejected': xxx},
            {'chosen': xxx, 'rejected': xxx},
         ]
    jsonl: {'chosen':
                [{'role': 'system', 'content': 'system_content'},
                {'role': 'user', 'content': 'user_content'},
                {'role': 'assistant', 'think': 'think_content', 'content': 'assistant_content'}],
            'rejected':
                [{'role': 'system', 'content': 'system_content'},
                {'role': 'user', 'content': 'user_content'},
                {'role': 'assistant', 'think': 'think_content', 'content': 'assistant_content'}],
            }\n
           {'chosen':
                [{'role': 'system', 'content': 'system_content'},
                {'role': 'user', 'content': 'user_content'},
                {'role': 'assistant', 'think': 'think_content', 'content': 'assistant_content'}],
            'rejected':
                [{'role': 'system', 'content': 'system_content'},
                {'role': 'user', 'content': 'user_content'},
                {'role': 'assistant', 'think': 'think_content', 'content': 'assistant_content'}],
            }
    pkl: [
            {'chosen': xxx, 'rejected': xxx},
            {'chosen': xxx, 'rejected': xxx},
         ]
    """
    def __init__(self, file_path, block_size):
        self.block_size = block_size
        self.data = []
        self.plain_text = False

        file_type = _get_file_type(file_path)

        if file_type == 'npy':
            try:
                self.data = np.load(file_path, mmap_mode='r')
            except ValueError:
                self.data = np.load(file_path, allow_pickle=True)
        elif file_type == 'jsonl':
            self.plain_text = True
            with open(file_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        elif file_type == 'pkl':
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type for {file_path}")

    def __len__(self):
        return len(self.data)

    def _split_prompt_completion(self, token_ids: list):
        """将序列按最后一个 <assistant> 切分为 (prompt_ids, completion_ids)"""
        assistant_id = TrainerTools().tokenizer.assistant

        last_assistant_idx = -1
        for idx in range(len(token_ids) - 1, -1, -1):
            if token_ids[idx] == assistant_id:
                last_assistant_idx = idx
                break

        if last_assistant_idx == -1:
            return None, None

        prompt_ids = token_ids[:last_assistant_idx + 1]
        completion_ids = token_ids[last_assistant_idx + 1:]

        if len(completion_ids) == 0:
            return None, None

        return prompt_ids, completion_ids

    def __getitem__(self, item):
        record = self.data[item]

        chosen_raw = record['chosen']
        rejected_raw = record['rejected']

        if self.plain_text:
            chosen_id = TrainerTools().tokenizer.apply_chat_template(chosen_raw)
            rejected_id = TrainerTools().tokenizer.apply_chat_template(rejected_raw)
        else:
            chosen_id = chosen_raw
            rejected_id = rejected_raw

        if isinstance(chosen_id, (np.ndarray, torch.Tensor)): chosen_id = chosen_id.tolist()
        if isinstance(rejected_id, (np.ndarray, torch.Tensor)): rejected_id = rejected_id.tolist()

        chosen_prompt, chosen_comp = self._split_prompt_completion(chosen_id)
        rejected_prompt, rejected_comp = self._split_prompt_completion(rejected_id)

        user_id = TrainerTools().tokenizer.user
        system_id = TrainerTools().tokenizer.system
        assistant_id = TrainerTools().tokenizer.assistant

        # 如果未找到 assistant 标记或无有效回答，将 [user_id, assistant_id] 置于序列开头，确保后续 token 作为回答保留
        if chosen_prompt is None or rejected_prompt is None:
            max_body = max(self.block_size - 2, 1)
            safe_chosen = ([user_id, assistant_id] + chosen_id[:max_body]) if chosen_prompt is None else chosen_id[
                                                                                                         :self.block_size]
            safe_rejected = (
                        [user_id, assistant_id] + rejected_id[:max_body]) if rejected_prompt is None else rejected_id[
                                                                                                          :self.block_size]
            return {
                'chosen': safe_chosen,
                'rejected': safe_rejected
            }

        min_prompt_len = 2
        max_comp_len = max(self.block_size - min_prompt_len, 1)

        if len(chosen_comp) > max_comp_len:
            chosen_comp = chosen_comp[:max_comp_len]
        if len(rejected_comp) > max_comp_len:
            rejected_comp = rejected_comp[:max_comp_len]

        avail_prompt_len = max(self.block_size - max(len(chosen_comp), len(rejected_comp)), min_prompt_len)

        def _truncate_prompt(p_ids: list) -> list:
            res = list(p_ids)
            if len(res) > avail_prompt_len:
                res = res[-avail_prompt_len:]
                if res[0] not in (user_id, system_id) and len(res) >= 2:
                    res[0] = user_id
                if res[-1] != assistant_id:
                    res[-1] = assistant_id
            return res

        c_prompt_ids = _truncate_prompt(chosen_prompt)
        r_prompt_ids = c_prompt_ids if chosen_prompt == rejected_prompt else _truncate_prompt(rejected_prompt)

        return {
            'chosen': c_prompt_ids + chosen_comp,
            'rejected': r_prompt_ids + rejected_comp
        }


class RLDataset(Dataset):
    """
        适用于RL阶段（例如：PPO、GRPO、GSPO），数据格式支持jsonl和pkl，如果是jsonl，则会在getitem阶段encode成token
        npy: [
                {'prompt': xxx, 'answer': xxx},
                {'prompt': xxx, 'answer': xxx},
             ]
        jsonl: {'prompt':
                    [{'role': 'system', 'content': 'system_content'},
                    {'role': 'user', 'content': 'user_content'}]
                'answer': '10'
               }\n
               {'prompt':
                    [{'role': 'system', 'content': 'system_content'},
                    {'role': 'user', 'content': 'user_content'}]
                'answer': '10'
               }
        pkl: [
                {'prompt': xxx, 'answer': xxx},
                {'prompt': xxx, 'answer': xxx},
             ]
        """
    def __init__(self, file_path):
        self.data = []
        self.plain_text = False

        file_type = _get_file_type(file_path)

        if file_type == 'npy':
            try:
                self.data = np.load(file_path, mmap_mode='r')
            except ValueError:
                self.data = np.load(file_path, allow_pickle=True)
        elif file_type == 'jsonl':
            self.plain_text = True

            with open(file_path, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line.strip()))
        elif file_type == 'pkl':
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type for {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        record = self.data[item]

        prompt_raw = record['prompt']
        answer_raw = record.get('answer', None)

        if self.plain_text:
            question = TrainerTools().tokenizer.apply_chat_template(prompt_raw)
            answer = TrainerTools().tokenizer.encode(answer_raw) if answer_raw else None
        else:
            question = prompt_raw
            answer = answer_raw

        # 转换为 Tensor
        if isinstance(question, np.ndarray):
            prompt_tensor = torch.from_numpy(question.astype(np.int64))
        else:
            prompt_tensor = torch.tensor(question).long()

        if prompt_tensor.numel() > 0 and prompt_tensor[-1] != TrainerTools().tokenizer.assistant:
            prompt_tensor = torch.cat([prompt_tensor, torch.tensor([TrainerTools().tokenizer.assistant], dtype=prompt_tensor.dtype)])

        if answer is not None:
            if isinstance(answer, np.ndarray):
                answer_tensor = torch.from_numpy(answer.astype(np.int64))
            else:
                answer_tensor = torch.tensor(answer).long()
        else:
            answer_tensor = None

        return {
            'prompt': prompt_tensor,
            'answer': answer_tensor
        }

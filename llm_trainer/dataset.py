import torch
from torch.utils.data import Dataset
import pickle
import csv
import json

from .tools import TrainerTools
from .utils import repeat_image_tok


"""
support jsonl and pkl
"""
def _get_file_type(file_path: str):
    if file_path.endswith('.jsonl'):
        return 'jsonl'
    elif file_path.endswith('.pkl'):
        return 'pkl'

    return None


class TextDataset(Dataset):
    """
    适用于pretrain阶段，数据格式支持jsonl和pkl，如果是jsonl会在init阶段全部encode成token
    jsonl: {'text': 'text1'}\n{'text': 'text2'}
    pkl: [0, 1, 2, 3 ...]
    """
    def __init__(
            self,
            file_path,
            block_size,
            stride
    ):
        super().__init__()

        self.input_ids = []

        file_type = _get_file_type(file_path)
        if file_type == 'jsonl':
            tokens = []
            with open(file_path, 'r') as f:
                for line in f:
                    tokens.extend(TrainerTools().tokenizer.encode(json.loads(line.strip())['text']))
        elif file_type == 'pkl':
            with open(file_path, 'rb') as f:
                tokens = pickle.load(f)
        else:
            raise Exception(f'unsupported file type for {file_path}')

        for i in range(0, len(tokens) - block_size + 1, stride):
            self.input_ids.append(tokens[i:i+block_size])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return torch.tensor(self.input_ids[item]).long()


class LineByLineTextDataset(Dataset):
    """
    适用于sft阶段，数据格式支持jsonl和pkl，如果是jsonl，则会在getitem阶段encode成token
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
            max_len,
            image_tags_file_path=None,
            tokens_per_image=-1
    ):
        super().__init__()

        self.max_len = max_len
        self.tokens_per_image = tokens_per_image
        self.input_ids = []
        self.image_tags = []
        self.plain_text = False

        file_type = _get_file_type(file_path)
        if file_type == 'jsonl':
            self.plain_text = True

            with open(file_path, 'r') as f:
                for line in f:
                    self.input_ids.append(json.loads(line.strip()))
        elif file_type == 'pkl':
            with open(file_path, 'rb') as f:
                self.input_ids = pickle.load(f)
        else:
            raise Exception(f'unsupported file type for {file_path}')

        if image_tags_file_path:
            with open(image_tags_file_path, 'r') as f:
                csv_reader = csv.reader(f)
                for line in csv_reader:
                    self.image_tags.append(line[0])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        if self.plain_text:
            inputs = TrainerTools().tokenizer.apply_chat_template(self.input_ids[item])
        else:
            inputs = self.input_ids[item]

        inputs = torch.tensor(inputs).long()
        image_tag = self.image_tags[item] if self.image_tags else None

        if self.tokens_per_image != -1:
            inputs = repeat_image_tok(inputs, self.tokens_per_image)
        else:
            image_tag = None

        inputs = inputs[:self.max_len]

        return {
            'inputs': inputs,
            'image_tag': image_tag
        }


class DPODataset(Dataset):
    """
    适用于dpo阶段，数据格式支持jsonl和pkl，如果是jsonl，则会在getitem阶段encode成token
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
                'role': 'assistant', 'think': 'think_content', 'content': 'assistant_content'}],
            }
    pkl: [
            {'chosen': xxx, 'rejected': xxx},
            {'chosen': xxx, 'rejected': xxx},
         ]
    """
    def __init__(self, file_path, max_len):
        self.max_len = max_len
        self.chosen_ids = []
        self.rejected_ids = []
        self.plain_text = False

        file_type = _get_file_type(file_path)
        if file_type == 'jsonl':
            self.plain_text = True

            with open(file_path, 'r') as f:
                for line in f:
                    json_ = json.loads(line.strip())
                    self.chosen_ids.append(json_['chosen'])
                    self.rejected_ids.append(json_['rejected'])
        elif file_type == 'pkl':
            with open(file_path, 'rb') as f:
                tokens = pickle.load(f)

            for token in tokens:
                self.chosen_ids.append(token['chosen'])
                self.rejected_ids.append(token['rejected'])
        else:
            raise Exception(f'unsupported file type for {file_path}')

    def __len__(self):
        return len(self.chosen_ids)

    def __getitem__(self, item):
        if self.plain_text:
            chosen_id = TrainerTools().tokenizer.apply_chat_template(self.chosen_ids[item])
            rejected_id = TrainerTools().tokenizer.apply_chat_template(self.rejected_ids[item])
        else:
            chosen_id = self.chosen_ids[item]
            rejected_id = self.rejected_ids[item]

        return {
            'chosen': chosen_id[:self.max_len],
            'rejected': rejected_id[:self.max_len]
        }


class GRPORolloutDataset(Dataset):
    """
        适用于grpo(gspo)阶段，数据格式支持jsonl和pkl，如果是jsonl，则会在getitem阶段encode成token
        jsonl: {'prompt':
                    [{'role': 'system', 'content': 'system_content'},
                    {'role': 'user', 'content': 'user_content'},
                    {'role': 'assistant', 'think': 'think_content', 'content': 'assistant_content'}],
                'answer': '10'
               }\n
               {'prompt':
                    [{'role': 'system', 'content': 'system_content'},
                    {'role': 'user', 'content': 'user_content'},
                    {'role': 'assistant', 'think': 'think_content', 'content': 'assistant_content'}],
                'answer': '10'
               }
        pkl: [
                {'prompt': xxx, 'answer': xxx},
                {'prompt': xxx, 'answer': xxx},
             ]
        """
    def __init__(self, file_path):
        self.questions = []
        self.answers = []
        self.plain_text = False

        file_type = _get_file_type(file_path)
        if file_type == 'jsonl':
            self.plain_text = True

            with open(file_path, 'r') as f:
                for line in f:
                    json_ = json.loads(line.strip())
                    self.questions.append(json_['prompt'])
                    self.answers.append(json_['answer'])
        elif file_type == 'pkl':
            with open(file_path, 'rb') as f:
                tokens = pickle.load(f)

            for token in tokens:
                self.questions.append(token['prompt'])
                self.answers.append(token['answer'])
        else:
            raise Exception(f'unsupported file type for {file_path}')

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        if self.plain_text:
            question = TrainerTools().tokenizer.apply_chat_template(self.questions[item])
            answer = TrainerTools().tokenizer.encode(self.answers[item])
        else:
            question = self.questions[item]
            answer = self.answers[item]

        return {
            'prompt': torch.tensor(question).long(),
            'answer': torch.tensor(answer).long()
        }

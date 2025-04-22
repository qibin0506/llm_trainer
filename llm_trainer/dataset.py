import os.path

import torch
from torch.utils.data import Dataset
import pickle

from .tools import TrainerTools
from .utils import extra_image_tag_and_repeat_image_tok


def _try_load_pkl(file_path: str):
    tokens = None
    try:
        with open(file_path, 'rb') as f:
            tokens = pickle.load(f)
    except Exception as e:
        raise e
    finally:
        return tokens


class TextDataset(Dataset):
    """
    适用于pretrain阶段
    """
    def __init__(self, file_path, block_size, stride):
        super().__init__()

        self.input_ids = []

        tokens = _try_load_pkl(file_path)
        if not tokens:
            cache_file = f'{file_path}.cache'
            if os.path.exists(cache_file):
                tokens = _try_load_pkl(cache_file)
            else:
                with open(file_path, 'r') as f:
                    tokens = TrainerTools().tokenizer.encode_to_token(f.read(), False, covert_tensor=False)

                with open(cache_file, 'wb') as f:
                    pickle.dump(tokens, f)

        for i in range(0, len(tokens) - block_size + 1, stride):
            self.input_ids.append(tokens[i:i+block_size])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return torch.tensor(self.input_ids[item]).long()


class LineByLineTextDataset(Dataset):
    """
    适用于sft阶段
    """
    def __init__(self, file_path, max_len, tokens_per_image=-1):
        super().__init__()

        self.max_len = max_len
        self.tokens_per_image = tokens_per_image
        self.input_ids = []

        tokens = _try_load_pkl(file_path)
        if not tokens:
            cache_file = f'{file_path}.cache'
            if os.path.exists(cache_file):
                tokens = _try_load_pkl(cache_file)
            else:
                tokens = []
                with open(file_path, 'r') as f:
                    for line in f:
                        tokens.append(TrainerTools().tokenizer.encode_to_token(line, False, covert_tensor=False))

                with open(cache_file, 'wb') as f:
                    pickle.dump(tokens, f)

        self.input_ids = tokens

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        inputs = self.input_ids[item]
        if self.tokens_per_image != -1:
            inputs, image_tag = extra_image_tag_and_repeat_image_tok(inputs, self.tokens_per_image)
        else:
            image_tag = None

        inputs = inputs[:self.max_len]

        return {'inputs': torch.tensor(inputs).long(), 'image_tag': image_tag}


class DPODataset(Dataset):
    def __init__(self, file_path, max_len):
        self.max_len = max_len
        self.chosen_ids = []
        self.rejected_ids = []

        # [{'chosen': xxx, 'rejected': xxx} ...]
        tokens = _try_load_pkl(file_path)
        for token in tokens:
            self.chosen_ids.append(token['chosen'])
            self.rejected_ids.append(token['rejected'])

    def __len__(self):
        return len(self.chosen_ids)

    def __getitem__(self, item):
        chosen_id = self.chosen_ids[item]
        rejected_id = self.rejected_ids[item]

        return {'chosen': chosen_id[:self.max_len], 'rejected': rejected_id[:self.max_len]}


class GRPORolloutDataset(Dataset):
    def __init__(self, file_path):
        self.questions = []
        self.answers = []

        # [{'question': xxx, 'answer': ''}]
        tokens = _try_load_pkl(file_path)
        for token in tokens:
            self.questions.append(token['prompt'])
            self.answers.append(token['answer'])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = self.questions[item]
        answer = self.answers[item]

        return {
            'prompt': torch.tensor(question).long(),
            'answer': torch.tensor(answer).long()
        }

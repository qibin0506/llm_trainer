import torch
from torch.utils.data import Dataset
import pickle


class TextDataset(Dataset):
    """
    适用于pretrain阶段
    """
    def __init__(self, file_path, block_size, stride):
        super().__init__()

        self.input_ids = []

        with open(file_path, 'rb') as f:
            all_tokens = pickle.load(f)

        for i in range(0, len(all_tokens) - block_size, stride):
            self.input_ids.append(all_tokens[i:i+block_size])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return torch.tensor(self.input_ids[item]).long()


class LineByLineTextDataset(Dataset):
    """
    适用于sft阶段
    """
    def __init__(self, file_path):
        super().__init__()

        self.input_ids = []
        with open(file_path, 'rb') as f:
            tokens = pickle.load(f)
            self.input_ids = tokens

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        # inputs = inputs[:self.ctx_len]
        return torch.tensor(self.input_ids[item]).long()

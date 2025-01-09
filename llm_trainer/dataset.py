import torch
from torch.utils.data import Dataset
import pickle


class BaseDataset(Dataset):
    def get_first(self):
        raise NotImplementedError()


class LLMDataset(BaseDataset):
    def __init__(self, file_path):
        super().__init__()
        # self.ctx_len = ctx_len

        self.tokens = []
        with open(file_path, 'rb') as f:
            tokens = pickle.load(f)
            self.tokens.extend(tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        inputs = self.tokens[item]
        # inputs = inputs[:self.ctx_len]
        return torch.tensor(inputs).long()

    def get_first(self):
        inputs = self.tokens[0]
        # inputs = inputs[:self.ctx_len]
        return torch.tensor(inputs).long()
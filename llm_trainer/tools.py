import os
from abc import ABC, abstractmethod
import torch
from .tokenizer import Tokenizer
from .parallel import DsParallel, DdpParallel, NoneParallel
from .log import Logger


parallel_types = {
    'ds': DsParallel,
    'ddp': DdpParallel,
    'none': NoneParallel
}

dtypes = {
    'float': torch.float,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64
}

class TrainerTools:
    def __init__(self):
        if not hasattr(TrainerTools, "_first_init"):
            TrainerTools._first_init = True

            self.parallel = self._new_parallel()

            self.tokenizer = Tokenizer()
            self.use_amp = 'cuda' in self.parallel.device and not isinstance(self.parallel, DsParallel)

            Logger.std_log(f'word_size={self.parallel.world_size}, use_amp={self.use_amp}')

    def _new_parallel(self):
        parallel_type = os.environ.get('PARALLEL_TYPE', 'none')
        Logger.std_log(f'parallel_type={parallel_type}')
        return parallel_types[parallel_type]()

    def __new__(cls, *args, **kwargs):
        if not hasattr(TrainerTools, "_instance"):
            TrainerTools._instance = object.__new__(cls)

        return TrainerTools._instance


class FileDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx) -> str: ...


def estimate_data_size(
        file_dataset: FileDataset,
        max_seq_len: int,
        type: str
) -> int:
    """
    估计数据集大小
    """
    data_size = 0
    files_count = len(file_dataset)

    if type == 'sft':
        from .dataset import SFTDataset
        for idx in range(files_count):
            dataset = SFTDataset(file_dataset[idx], max_seq_len)
            data_size += len(dataset)
    elif type == 'dpo':
        from .dataset import DPODataset
        for idx in range(files_count):
            dataset = DPODataset(file_dataset[idx], max_seq_len)
            data_size += len(dataset)
    elif type == 'grpo' or type == 'ppo':
        from .dataset import RLDataset
        for idx in range(files_count):
            dataset = RLDataset(file_dataset[idx])
            data_size += len(dataset)
    else:
        from .dataset import PretrainDataset
        for idx in range(files_count):
            dataset = PretrainDataset(
                file_dataset[idx],
                max_seq_len,
                max_seq_len
            )
            data_size += len(dataset)

    return data_size


def extract_policy_weights_from_ppo(model_config, ppo_weights):
    from llm_model import LlmModel
    from .ppo_trainer import PolicyAndValueModelWrapper, ValueModel

    policy_model = LlmModel(model_config)
    value_model = ValueModel(LlmModel(model_config))

    wrapper = PolicyAndValueModelWrapper(policy_model, value_model)
    wrapper.load_state_dict(ppo_weights)

    return wrapper.policy_model.state_dict()


def extract_value_weights_from_ppo(model_config, ppo_weights):
    from llm_model import LlmModel
    from .ppo_trainer import PolicyAndValueModelWrapper, ValueModel

    policy_model = LlmModel(model_config)
    value_model = ValueModel(LlmModel(model_config))

    wrapper = PolicyAndValueModelWrapper(policy_model, value_model)
    wrapper.load_state_dict(ppo_weights)

    return wrapper.value_model.state_dict()
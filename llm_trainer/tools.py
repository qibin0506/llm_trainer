import os
import torch
from .tokenizer import Tokenizer
from .parallel_ds import DsParallel
from .parallel_fsdp import FsdpParallel
from .parallel_ddp import DdpParallel
from .parallel_none import NoneParallel
from .log import log


parallel_types = {
    'ds': DsParallel,
    'fsdp': FsdpParallel,
    'ddp': DdpParallel,
    'none': NoneParallel
}

class TrainerTools:
    def __init__(self):
        if not hasattr(TrainerTools, "_first_init"):
            TrainerTools._first_init = True

            self.parallel = self.new_parallel()

            self.tokenizer = Tokenizer(os.environ.get('TOKENIZERS_TYPE', 'zh_llama'))
            self.use_amp = 'cuda' in self.parallel.device and not isinstance(self.parallel, DsParallel)
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            log(f'word_size={self.parallel.world_size},'
                f' use_amp={self.use_amp},'
                f' dtype={self.dtype}')

    def new_parallel(self):
        parallel_type = os.environ.get('PARALLEL_TYPE', 'none')
        log(f'parallel_type={parallel_type}')
        return parallel_types[parallel_type]()

    def __new__(cls, *args, **kwargs):
        if not hasattr(TrainerTools, "_instance"):
            TrainerTools._instance = object.__new__(cls)

        return TrainerTools._instance


def estimate_data_size(
        all_files: list[str],
        max_position_embeddings: int,
        type: str
) -> int:
    """
    估计数据集大小
    """
    data_size = 0

    if type == 'sft':
        from .dataset import LineByLineTextDataset
        for file_path in all_files:
            dataset = LineByLineTextDataset(file_path, max_position_embeddings)
            data_size += len(dataset)
    elif type == 'dpo':
        from .dataset import DPODataset
        for file_path in all_files:
            dataset = DPODataset(file_path, max_position_embeddings)
            data_size += len(dataset)
    elif type == 'grpo':
        from .dataset import GRPORolloutDataset
        for file_path in all_files:
            dataset = GRPORolloutDataset(file_path)
            data_size += len(dataset)
    else:
        from .dataset import TextDataset
        for file_path in all_files:
            dataset = TextDataset(
                file_path,
                max_position_embeddings,
                max_position_embeddings
            )
            data_size += len(dataset)

    return data_size
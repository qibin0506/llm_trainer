import os
import torch
from .tokenizer import Tokenizer
from .parallel import Parallel
from .parallel_fsdp import FsdpParallel
from .parallel_ddp import DdpParallel
from .parallel_none import NoneParallel
from .log import log


class TrainerTools:
    def __init__(self):
        if not hasattr(TrainerTools, "_first_init"):
            TrainerTools._first_init = True

            parallel_type = os.environ.get('PARALLEL_TYPE', 'none')
            if parallel_type == 'fsdp':
                self.parallel: Parallel = FsdpParallel()
            elif parallel_type == 'ddp':
                self.parallel: Parallel = DdpParallel()
            else:
                self.parallel: Parallel = NoneParallel()

            self.tokenizer = Tokenizer(os.environ.get('TOKENIZERS_TYPE', 'bert'))
            self.use_amp = 'cuda' in self.parallel.device
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            log(f'word_size={self.parallel.world_size}, use_amp={self.use_amp}, dtype={self.dtype}')

    def __new__(cls, *args, **kwargs):
        if not hasattr(TrainerTools, "_instance"):
            TrainerTools._instance = object.__new__(cls)

        return TrainerTools._instance


def estimate_data_size(
        all_files: list[str],
        max_position_embeddings: int,
        is_sft: bool
) -> int:
    """
    估计数据集大小
    """
    from .dataset import TextDataset, LineByLineTextDataset
    data_size = 0

    if not is_sft:
        for file_path in all_files:
            dataset = TextDataset(
                file_path,
                max_position_embeddings,
                max_position_embeddings
            )
            data_size += len(dataset)
    else:
        for file_path in all_files:
            dataset = LineByLineTextDataset(file_path)
            data_size += len(dataset)

    return data_size
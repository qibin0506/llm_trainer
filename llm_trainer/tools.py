import os
import torch
from .tokenizer import Tokenizer
from .parallel_ds import DsParallel
from .parallel_fsdp import FsdpParallel
from .parallel_ddp import DdpParallel
from .parallel_none import NoneParallel
from .log import log


class TrainerTools:
    def __init__(self):
        if not hasattr(TrainerTools, "_first_init"):
            TrainerTools._first_init = True

            parallel_types = {
                'ds': DsParallel,
                'fsdp': FsdpParallel,
                'ddp': DdpParallel,
                'none': NoneParallel
            }

            parallel_type = os.environ.get('PARALLEL_TYPE', 'none')

            # smart模式
            # 如果多卡
            #   1. 安装了deepspeed，则使用deepspeed
            #   2. 没有安装deepspeed, 使用ddp
            # 否则使用none
            if parallel_type == 'smart':
                global_rank = int(os.environ.get('RANK', -1))
                if global_rank == -1:
                    parallel_type = 'none'
                else:
                    try:
                        import deepspeed
                        parallel_type = 'ds'
                    except:
                        parallel_type = 'ddp'

            self.parallel = parallel_types[parallel_type]()

            self.tokenizer = Tokenizer(os.environ.get('TOKENIZERS_TYPE', 'qwen'))
            self.use_amp = 'cuda' in self.parallel.device and not isinstance(self.parallel, DsParallel)
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            log(f'parallel_type={parallel_type},'
                f' word_size={self.parallel.world_size},'
                f' use_amp={self.use_amp},'
                f' dtype={self.dtype}')

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
import os
import torch
from .tokenizer import Tokenizer
from .parallel import Parallel
from .parallel_fsdp import FsdpParallel
from .parallel_ddp import DdpParallel
from .parallel_none import NoneParallel


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

            self.bot_token = self.tokenizer.encode_to_token('[BOT]', unsqueeze=False, covert_tensor=False)[0]

    def __new__(cls, *args, **kwargs):
        if not hasattr(TrainerTools, "_instance"):
            TrainerTools._instance = object.__new__(cls)

        return TrainerTools._instance
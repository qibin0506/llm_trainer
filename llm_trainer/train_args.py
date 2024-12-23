from typing import Optional
from dataclasses import dataclass
from torch import nn
from llama import LlamaConfig


@dataclass
class FsdpArgs:
    transformer_layer_cls: Optional[nn.Module] = None,
    wrap_policy_num_params: int = -1
    cpu_offload: bool = False
    offload_params: bool = False


@dataclass
class DataLoaderArgs:
    data_loader_pin_memory: bool = False
    data_loader_num_workers: int = 0
    data_loader_shuffle: bool = False
    data_loader_drop_last: bool = True

@dataclass
class TrainArgs:
    n_epochs: int
    batch_size: int
    llama_config: LlamaConfig
    is_sft: bool
    all_data_size: int
    all_files: list[any]
    gradient_accumulation_steps: int = 0
    fsdp_args: FsdpArgs = FsdpArgs()
    data_loader_args: DataLoaderArgs = DataLoaderArgs()


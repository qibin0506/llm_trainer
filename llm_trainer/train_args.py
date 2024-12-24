from typing import Optional, Set, Type
from dataclasses import dataclass, field
from torch import nn
from llama import LlamaConfig


@dataclass
class FsdpArgs:
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
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
class LrSchedulerArgs:
    initial_lr: float = 1e-4
    max_lr: float = 5e-4
    warmup_iters_ratio: float = 0.2
    min_lr_ratio: float = 0.1


@dataclass
class TrainArgs:
    n_epochs: int
    batch_size: int
    llama_config: LlamaConfig
    is_sft: bool
    all_data_size: int
    all_files: list[any] = field(default_factory=list)
    gradient_accumulation_steps: int = 0
    lr_scheduler_args: LrSchedulerArgs = field(default_factory=LrSchedulerArgs)
    fsdp_args: FsdpArgs = field(default_factory=FsdpArgs)
    data_loader_args: DataLoaderArgs = field(default_factory=DataLoaderArgs)


from typing import Optional, Set, Type, Callable
from dataclasses import dataclass, field
from torch import nn
from llama import LlamaConfig


@dataclass
class FsdpArgs:
    """
    fsdp训练模式配置项
    Args:
        transformer_layer_cls (`Set[Type[nn.Module]]`, *optional*, default is None):
            提供transformer层的类
        wrap_policy_num_params (`int`, *optional*, default is -1)
            size_based_auto_wrap_policy的min_num_params参数，-1不生效该策略
        cpu_offload (`bool`, *optional*, default is False):
            是否使用cpu卸载
        offload_params (`bool`, default is False)
            是否卸载参数，在cpu_offload为True时生效
    """
    transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
    wrap_policy_num_params: int = -1
    cpu_offload: bool = False
    offload_params: bool = False


@dataclass
class DataLoaderArgs:
    """
    data loader配置项
    Args:
        data_loader_pin_memory (`bool`, *optional*, default is None):
            data_loader pin_memory config
        data_loader_num_workers (`int`, *optional*, default is 0)
            data_loader num_workers config
        data_loader_shuffle (`bool`, *optional*, default is False):
            是否需要shuffle数据
        data_loader_drop_last (`bool`, default is False)
            最后一个batch不满足batch_size时，是否丢弃
    """
    data_loader_pin_memory: bool = False
    data_loader_num_workers: int = 0
    data_loader_shuffle: bool = False
    data_loader_drop_last: bool = True

@dataclass
class LrSchedulerArgs:
    """
    lr scheduler配置项
    Args:
        enable_lr_scheduler (`bool`, default is True)
            是否启动lr scheduler
        initial_lr (`float`, *optional*, default is 1e-4):
            初始化lr
        max_lr (`float`, *optional*, default is 5e-4)
            最大lr
        warmup_iters_ratio (`float`, *optional*, default is 0.2):
            使用warmup的最大batch
        min_lr_ratio (`float`, default is 0.1)
            最小lr的比例，最小lr=min_lr_ratio*initial_lr
    """
    enable_lr_scheduler: bool = True
    initial_lr: float = 1e-4
    max_lr: float = 5e-4
    warmup_iters_ratio: float = 0.2
    min_lr_ratio: float = 0.1

@dataclass
class KDArgs:
    """
    知识蒸馏模式配置项
    Args:
        teacher_logits_provider (`Callable[..., nn.Module]`):
            知识蒸馏教师模型logits的提供者
        kd_coef (`float`, *optional*, default is 0.4)
            蒸馏loss的占比，loss = kd_coef * kd_loss + (1 - kd_coef) * lm_loss
    """
    teacher_logits_provider: Callable[..., nn.Module]
    kd_coef: float = 0.4


@dataclass
class TrainArgs:
    """
    训练参数配置项
    Args:
        n_epochs (`int`):
            训练epochs
        batch_size (`int`):
            每个batch的大小
        llama_config (`LlamaConfig`)
            llama模型的配置
        is_sft (`bool`)
            是否sft训练
        all_data_size (`int`)
            所有训练数据大小
        all_files (`list`)
            所有训练文件
        gradient_accumulation_steps (`int`, *Optional*, default is 0)
            梯度累积步数，为0时不使用梯度累积
        lr_scheduler_args (`LrSchedulerArgs`)
            lr scheduler配置项
        fsdp_args: (`FsdpArgs`)
            fsdp训练模式配置项
        data_loader_args: (`DataLoaderArgs`)
            data loader配置项
        kd_args: (`KDArgs`, *Optional*, default is None)
            知识蒸馏配置项，为None时不使用知识蒸馏
    """
    n_epochs: int
    batch_size: int
    llama_config: LlamaConfig
    is_sft: bool
    all_data_size: int
    all_files: list[any] = field(default_factory=list)
    gradient_accumulation_steps: int = 0
    lr_scheduler_args: LrSchedulerArgs = field(default_factory=LrSchedulerArgs)
    fsdp_args: FsdpArgs = field(default_factory=FsdpArgs)
    data_loader_args: DataLoaderArgs = field(default_factory=DataLoaderArgs),
    kd_args: Optional[KDArgs] = None


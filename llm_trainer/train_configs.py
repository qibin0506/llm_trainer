from typing import Optional, Union, Set, Type, Callable
from torch import nn
from llama import LlamaConfig

class DsOffloadConfig:
    def __init__(
            self,
            *,
            device: str = 'cpu',
            pin_memory: bool = False
    ):
        self.device = device
        self.pin_memory = pin_memory


class DsActivationCheckpointingConfig:
    def __init__(
            self,
            *,
            partition_activations: bool = False,
            cpu_checkpointing: bool = False,
            contiguous_memory_optimization: bool = False,
            number_checkpoints: Optional[int] = None,
            synchronize_checkpoint_boundary: bool = False,
            profile: bool = False
    ):
        self.partition_activations =partition_activations
        self.cpu_checkpointing = cpu_checkpointing
        self.contiguous_memory_optimization = contiguous_memory_optimization
        self.number_checkpoints = number_checkpoints
        self.synchronize_checkpoint_boundary = synchronize_checkpoint_boundary
        self.profile = profile


class DsZeROConfig:
    def __init__(
            self,
            *,
            stage: int,
            allgather_partitions: Optional[bool] = True,
            allgather_bucket_size: Optional[int] = 5e8,
            overlap_comm: Optional[bool] = True,
            reduce_scatter: Optional[bool] = True,
            reduce_bucket_size: Optional[Union[str, int]] = 5e8,
            contiguous_gradients: Optional[bool] = True
    ):
        self.stage = stage
        self.allgather_partitions = allgather_partitions
        self.allgather_bucket_size = allgather_bucket_size
        self.overlap_comm = overlap_comm
        self.reduce_scatter = reduce_scatter
        self.reduce_bucket_size = reduce_bucket_size
        self.contiguous_gradients = contiguous_gradients


class DsZero1Config(DsZeROConfig):
    def __init__(
            self,
            *,
            allgather_partitions: Optional[bool] = True,
            allgather_bucket_size: Optional[int] = 5e8,
            overlap_comm: Optional[bool] = True,
            reduce_scatter: Optional[bool] = True,
            reduce_bucket_size: Optional[Union[str, int]] = 5e8,
            contiguous_gradients: Optional[bool] = True
    ):
        super().__init__(
            stage=1,
            allgather_partitions=allgather_partitions,
            allgather_bucket_size=allgather_bucket_size,
            overlap_comm=overlap_comm,
            reduce_scatter=reduce_scatter,
            reduce_bucket_size=reduce_bucket_size,
            contiguous_gradients=contiguous_gradients
        )

class DsZero2Config(DsZeROConfig):
    def __init__(
            self,
            *,
            allgather_partitions: Optional[bool] = True,
            allgather_bucket_size: Optional[int] = 5e8,
            overlap_comm: Optional[bool] = True,
            reduce_scatter: Optional[bool] = True,
            reduce_bucket_size: Optional[Union[str, int]] = 5e8,
            contiguous_gradients: Optional[bool] = True,
            offload_optimizer: Optional[DsOffloadConfig] = DsOffloadConfig(),
            offload_param: Optional[DsOffloadConfig] = DsOffloadConfig(),

    ):
        super().__init__(
            stage=2,
            allgather_partitions=allgather_partitions,
            allgather_bucket_size=allgather_bucket_size,
            overlap_comm=overlap_comm,
            reduce_scatter=reduce_scatter,
            reduce_bucket_size=reduce_bucket_size,
            contiguous_gradients=contiguous_gradients
        )

        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param


class DsZero3Config(DsZeROConfig):
    def __init__(
            self,
            *,
            allgather_partitions: Optional[bool] = None,
            allgather_bucket_size: Optional[bool] = None,
            overlap_comm: Optional[bool] = True,
            reduce_scatter: Optional[bool] = None,
            reduce_bucket_size: Optional[Union[str, int]] = 'auto',
            contiguous_gradients: Optional[bool] = True,
            sub_group_size: Optional[int] = 1e9,
            stage3_prefetch_bucket_size: Optional[Union[str, int]] = 'auto',
            stage3_param_persistence_threshold: Optional[Union[str, int]] = 'auto',
            stage3_max_live_parameters: Optional[int] = 1e9,
            stage3_max_reuse_distance: Optional[int] = 1e9,
            stage3_gather_16bit_weights_on_model_save: Optional[bool] = True,
            offload_optimizer: Optional[DsOffloadConfig] = None,
            offload_param: Optional[DsOffloadConfig] = None,

    ):
        super().__init__(
            stage=3,
            allgather_partitions=allgather_partitions,
            allgather_bucket_size=allgather_bucket_size,
            overlap_comm=overlap_comm,
            reduce_scatter=reduce_scatter,
            reduce_bucket_size=reduce_bucket_size,
            contiguous_gradients=contiguous_gradients
        )

        self.sub_group_size = sub_group_size
        self.stage3_prefetch_bucket_size = stage3_prefetch_bucket_size
        self.stage3_param_persistence_threshold = stage3_param_persistence_threshold
        self.stage3_max_live_parameters = stage3_max_live_parameters
        self.stage3_max_reuse_distance = stage3_max_reuse_distance
        self.stage3_gather_16bit_weights_on_model_save = stage3_gather_16bit_weights_on_model_save

        self.offload_optimizer = offload_optimizer
        self.offload_param = offload_param

class DsFp16Config:
    """
        DeepSpeed fp16配置项
        参数说明：https://deepspeed.org.cn/docs/config-json/
    """
    def __init__(
            self,
            *,
            enabled: Union[str, bool] = 'auto',
            loss_scale: int = 0,
            loss_scale_window: int = 1000,
            initial_scale_power: int = 16,
            hysteresis: int = 2,
            min_loss_scale: int = 1,
            fp16_opt_level: Optional[str] = '02'
    ):
        self.enabled = enabled
        self.loss_scale = loss_scale
        self.loss_scale_window = loss_scale_window
        self.initial_scale_power = initial_scale_power
        self.hysteresis = hysteresis
        self.min_loss_scale = min_loss_scale
        self.fp16_opt_level = fp16_opt_level


class DsBf16Config:
    def __init__(
            self,
            *,
            enabled: bool = False
    ):
        self.enabled = enabled


class DsConfig:
    """
        DeepSpeed训练模式配置
    """
    def __init__(
            self,
            *,
            zero_config: Optional[DsZeROConfig] = DsZero3Config(),
            fp16_config: Optional[DsFp16Config] = DsFp16Config(),
            bf16_config: Optional[DsBf16Config] = DsBf16Config(),
            gradient_clipping: Optional[float] = 1.0,
            activation_checkpointing: Optional[DsActivationCheckpointingConfig] = None
    ):
        self.zero_config = zero_config
        self.fp16_config = fp16_config
        self.bf16_config = bf16_config
        self.gradient_clipping = gradient_clipping
        self.activation_checkpointing = activation_checkpointing


class FsdpConfig:
    """
        fsdp训练模式配置项
        Args:
            transformer_layer_cls (`Set[Type[nn.Module]]`, *optional*, default is None):
                提供transformer层的类
            wrap_policy_num_params (`int`, *optional*, default is -1):
                size_based_auto_wrap_policy的min_num_params参数，-1不生效该策略
            cpu_offload (`bool`, *optional*, default is False):
                是否使用cpu卸载
            offload_params (`bool`, default is False):
                是否卸载参数，在cpu_offload为True时生效
        """

    def __init__(
            self,
            *,
            transformer_layer_cls: Optional[Set[Type[nn.Module]]] = None,
            wrap_policy_num_params: int = -1,
            cpu_offload: bool = False,
            offload_params: bool = False,
    ):
        self.transformer_layer_cls = transformer_layer_cls
        self.wrap_policy_num_params = wrap_policy_num_params
        self.cpu_offload = cpu_offload
        self.offload_params = offload_params


class DataLoaderConfig:
    """
        data loader配置项
        Args:
            data_loader_pin_memory (`bool`, *optional*, default is None):
                data_loader pin_memory config
            data_loader_num_workers (`int`, *optional*, default is 0):
                data_loader num_workers config
            data_loader_shuffle (`bool`, *optional*, default is False):
                是否需要shuffle数据
            data_loader_drop_last (`bool`, default is False):
                最后一个batch不满足batch_size时，是否丢弃
        """

    def __init__(
            self,
            *,
            data_loader_pin_memory: bool = False,
            data_loader_num_workers: int = 0,
            data_loader_shuffle: bool = False,
            data_loader_drop_last: bool = True,
    ):
        self.data_loader_pin_memory = data_loader_pin_memory
        self.data_loader_num_workers = data_loader_num_workers
        self.data_loader_shuffle = data_loader_shuffle
        self.data_loader_drop_last = data_loader_drop_last


class LrSchedulerConfig:
    """
        lr scheduler配置项

        Args:
            enable_lr_scheduler (`bool`, default is True):
                是否启动lr scheduler
            initial_lr (`float`, *optional*, default is 1e-4):
                初始化lr
            max_lr (`float`, *optional*, default is 5e-4):
                最大lr
            warmup_iters_ratio (`float`, *optional*, default is 0.2):
                使用warmup的最大batch
            min_lr_ratio (`float`, default is 0.1):
                最小lr的比例，最小lr=min_lr_ratio*initial_lr
        """

    def __init__(
            self,
            *,
            enable_lr_scheduler: bool = True,
            initial_lr: float = 1e-4,
            max_lr: float = 5e-4,
            warmup_iters_ratio: float = 0.2,
            min_lr_ratio: float = 0.1,
    ):
        self.enable_lr_scheduler = enable_lr_scheduler
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.warmup_iters_ratio = warmup_iters_ratio
        self.min_lr_ratio = min_lr_ratio

class KDConfig:
    """
        知识蒸馏模式配置项

        Args:
            teacher_logits_provider (`Callable[..., nn.Module]`):
                知识蒸馏教师模型logits的提供者
            kd_coef (`float`, *optional*, default is 0.4):
                蒸馏loss的占比，loss = kd_coef * kd_loss + (1 - kd_coef) * lm_loss
        """

    def __init__(
            self,
            *,
            teacher_logits_provider: Callable[..., nn.Module],
            kd_coef: float = 0.4
    ):
        self.teacher_logits_provider = teacher_logits_provider
        self.kd_coef = kd_coef


class TrainConfig:
    """
        训练参数配置项

        Args:
            n_epochs (`int`):
                训练epochs
            batch_size (`int`):
                每个batch的大小
            llama_config (`LlamaConfig`):
                llama模型的配置
            all_data_size (`int`):
                所有训练数据大小
            all_files (`list`):
                所有训练文件
            gradient_accumulation_steps (`int`, *Optional*, default is 0):
                梯度累积步数，为0时不使用梯度累积
            eval_batch_interval (`int`, default is 100):
                每隔多少个batch进行模型eval
            lr_scheduler_config (`LrSchedulerConfig`):
                lr scheduler配置项
            fsdp_config: (`FsdpConfig`):
                fsdp训练模式配置项
            data_loader_config: (`DataLoaderConfig`):
                data loader配置项
            kd_config: (`KDConfig`, *Optional*, default is None):
                知识蒸馏配置项，为None时不使用知识蒸馏
        """

    def __init__(
            self,
            n_epochs: int,
            batch_size: int,
            *,
            llama_config: LlamaConfig,
            all_data_size: int,
            all_files: Optional[list[any]] = None,
            gradient_accumulation_steps: int = 0,
            eval_batch_interval: int = 100,
            lr_scheduler_config: LrSchedulerConfig = LrSchedulerConfig(),
            ds_config: DsConfig = DsConfig(),
            fsdp_config: FsdpConfig = FsdpConfig(),
            data_loader_config: DataLoaderConfig = DataLoaderConfig(),
            kd_config: Optional[KDConfig] = None
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.llama_config = llama_config
        self.all_data_size = all_data_size
        self.all_files = all_files
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_batch_interval = eval_batch_interval
        self.lr_scheduler_config = lr_scheduler_config
        self.ds_config = ds_config
        self.fsdp_config = fsdp_config
        self.data_loader_config = data_loader_config
        self.kd_config = kd_config



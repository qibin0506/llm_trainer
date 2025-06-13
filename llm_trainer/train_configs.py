from typing import Optional, Union, Set, Type, Callable, List, Mapping, Any

import torch
from torch import nn
from llm_model import ModelConfig, VLMConfig
from .tools import FileDataset


class DsOffloadConfig:
    def __init__(
            self,
            *,
            device: str = 'cpu',
            pin_memory: bool = True
    ):
        self.device = device
        self.pin_memory = pin_memory


class DsActivationCheckpointingConfig:
    def __init__(
            self,
            *,
            partition_activations: bool = True,
            cpu_checkpointing: bool = True,
            contiguous_memory_optimization: bool = True,
            number_checkpoints: Optional[int] = None,
            synchronize_checkpoint_boundary: bool = True,
            profile: bool = True
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
            offload_optimizer: Optional[DsOffloadConfig] = None,
            offload_param: Optional[DsOffloadConfig] = None,

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
            enabled: bool = True
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


class LrConfig:
    def __init__(
            self,
            *,
            enable_lr_scheduler: bool = False,
            initial_lr: Optional[float] = None,
            weight_decay: float = 0.1,
            max_lr: Optional[float] = None,
            min_lr: Optional[float] = None,
            period: Optional[int] = None,
            period_mul: Optional[int] = None,
            warmup_iters: Optional[int] = None
    ):
        self.enable_lr_scheduler = enable_lr_scheduler
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.period = period
        self.period_mul = period_mul
        self.warmup_iters = warmup_iters


class LossConfig:
    def __init__(
            self,
            *,
            critical_tokens: Optional[List[int]] = None,
            critical_alpha: float = 1.0,
            aux_loss_coef: Optional[float] = 1.0
    ):
        super().__init__()
        self.critical_tokens = critical_tokens
        self.critical_alpha = critical_alpha
        self.aux_loss_coef = aux_loss_coef


class DPOConfig:
    def __init__(
            self,
            loss_beta: float,
            loss_label_smoothing: float = 0.0,
            loss_ipo: bool = False,
            nll_loss_coef: Optional[float] = None
    ):
        super().__init__()
        self.loss_beta = loss_beta
        self.loss_label_smoothing = loss_label_smoothing
        self.loss_ipo = loss_ipo
        self.nll_loss_coef = nll_loss_coef


class GRPOConfig:
    def __init__(
            self,
            grpo_steps: int = 1,
            clip_eps: float = 0.2,
            kl_weight: float = 0.01,
            group_size: int = 12,
            gen_max_new_tokens: Optional[int] = None,
            gen_temperature: Optional[float] = None,
            gen_k: Optional[int] = None,
            gen_p: Optional[float] = None,
            gen_suppress_tokens: Optional[list[int]] = None,
    ):
        self.grpo_steps = grpo_steps
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight
        self.group_size = group_size
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_temperature = gen_temperature
        self.gen_k = gen_k
        self.gen_p = gen_p
        self.gen_suppress_tokens = gen_suppress_tokens


class KDConfig:
    """
        知识蒸馏模式配置项

        Args:
            teacher_logits_provider (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
                知识蒸馏教师模型logits的提供者
            kd_coef (`float`, *optional*, default is 0.4):
                蒸馏loss的占比，loss = kd_coef * kd_loss + (1 - kd_coef) * lm_loss
        """

    def __init__(
            self,
            *,
            teacher_logits_provider: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            kd_coef: float = 0.4
    ):
        self.teacher_logits_provider = teacher_logits_provider
        self.kd_coef = kd_coef


class EvalConfig:
    def __init__(
            self,
            max_new_tokens: int = 512,
            temperature: float = 1.0,
            top_p: float = 0.95,
            top_k: Optional[float] = None
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


class TrainConfig:
    """
        训练参数配置项

        Args:
            n_epochs (`int`):
                训练epochs
            batch_size (`int`):
                每个batch的大小
            model_config (`ModelConfig`):
                模型的配置
            file_dataset (`FileDataset`):
                训练文件dataset
            mask_prompt (`bool`)
                指定是否mask prompt部分的token
            gradient_accumulation_steps (`int`, *Optional*, default is 0):
                梯度累积步数，为0时不使用梯度累积
                grpo训练时不生效该配置！
            eval_batch_interval (`int`, default is 100):
                每隔多少个batch进行模型eval
            lr_config (`LrConfig`):
                lr配置项
            fsdp_config: (`FsdpConfig`):
                fsdp训练模式配置项
            data_loader_config: (`DataLoaderConfig`):
                data loader配置项
            kd_config: (`KDConfig`, *Optional*, default is None):
                知识蒸馏配置项，为None时不使用知识蒸馏
            pixel_values_provider: (`Callable[[list[str]], torch.Tensor]`, *Optional*, default is None):
                训练vlm时根据image_tag提供pixel_values信息
        """

    def __init__(
            self,
            n_epochs: int,
            batch_size: int,
            *,
            model_config: Union[ModelConfig, VLMConfig],
            file_dataset: FileDataset,
            image_tags_file_dataset: Optional[FileDataset] = None,
            mask_prompt: bool = True,
            gradient_accumulation_steps: int = 0,
            eval_batch_interval: int = 100,
            loss_config: LossConfig = LossConfig(),
            dpo_config: Optional[DPOConfig] = None,
            grpo_config: Optional[GRPOConfig] = None,
            lr_config: LrConfig = LrConfig(),
            ds_config: DsConfig = DsConfig(),
            fsdp_config: FsdpConfig = FsdpConfig(),
            data_loader_config: DataLoaderConfig = DataLoaderConfig(),
            kd_config: Optional[KDConfig] = None,
            pixel_values_provider: Optional[Callable[[list[str]], torch.Tensor]] = None,
            init_state_dict: Optional[Mapping[str, Any]] = None,
            eval_config: EvalConfig = EvalConfig()
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.model_config = model_config
        self.file_dataset = file_dataset
        self.image_tags_file_dataset = image_tags_file_dataset
        self.mask_prompt = mask_prompt
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.eval_batch_interval = eval_batch_interval
        self.loss_config = loss_config
        self.dpo_config = dpo_config
        self.grpo_config = grpo_config
        self.lr_config = lr_config
        self.ds_config = ds_config
        self.fsdp_config = fsdp_config
        self.data_loader_config = data_loader_config
        self.kd_config = kd_config
        self.pixel_values_provider = pixel_values_provider
        self.init_state_dict = init_state_dict
        self.eval_config = eval_config



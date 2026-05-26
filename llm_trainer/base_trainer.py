from typing import Optional, Tuple, List, Dict, Any, Callable
import os
import copy
import gc
import math
import importlib.metadata
from packaging import version
from itertools import islice

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from llm_model import (
    LlmModel,
    ModelConfig
)

from .parallel import DsParallel
from .tools import TrainerTools
from .loss import LMLoss, KDLoss
from .partition_utils import unwrap_model_for_generation
from .log import (
    Logger,
    _get_log_dir
)
from .train_configs import (
    TrainConfig,
    DsZero2Config,
    DsZero3Config,
    KDConfig,
    GenerateConfig
)
from .scheduler import (
    LRScheduler,
    WarmupCosineAnnealingLRScheduler,
    NoneLRScheduler
)
from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
    load_steps,
    save_steps,
)
from .utils import (
    default_seed,
    set_seed,
    autocast,
    is_bf16_supported,
    is_fp16_supported,
    empty_cache
)
from .generate_utils import generate


class BaseTrainer:
    """
        BaseTrainer

        Args:
            train_config:
                - 全局训练配置类，包含模型配置、优化器、调度器以及特定算法配置（如 DPO、PPO、GRPO 等）。

            eval_prompts:
                - 用于评估阶段生成测试的文本提示词列表。
                - 长度为 [num_eval_prompts] 的字符串列表。

            generation_service:
                - 外部自定义生成服务接口
                - 签名:
                    1. model (torch.nn.Module): 传入的正在执行训练的模型实例（可能已被 DeepSpeed 封装）。
                    2. prompts (torch.Tensor): 待生成的一组 Prompt 文本。Shape: [batch_size]。
                    3. config (GenerateConfig): 生成解码控制配置（如 temp, top_p, top_k 等）。
                    4. task_type (str): 调用任务上下文类型，如 'eval', 'ppo', 'grpo'。
                    5. pixel_values (Optional[torch.Tensor]): VLM 多模态特征张量。Shape: [batch_size, channels, height, width] 或 [batch_size * num_images, channels, height, width]。
                    6. tokens_per_image (Optional[int]): 每个图片标签对应的虚拟 Token 数值标量。
                - 返回值:
                    - List[List[int]]: 外层列表长度为 [batch_size * group_size]，内层为生成的 Completion Token ID 序列（不应包含 Prompt）。

            kd_config:
                - 知识蒸馏 (Knowledge Distillation) 配置类。

            gradient_accumulation_steps:
                - 梯度累积步数，用于通过累积多批数据的梯度来模拟更大的 Global Batch Size。
        """
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            generation_service: Optional[Callable[[torch.nn.Module, torch.Tensor, GenerateConfig, str, Optional[torch.Tensor], Optional[int]], List[List[int]]]] = None,
            kd_config: Optional[KDConfig] = None,
            gradient_accumulation_steps: int = 1
    ):
        set_seed(default_seed)

        self.is_ds = isinstance(TrainerTools().parallel, DsParallel)
        self.train_config: TrainConfig = train_config
        self.eval_prompts = eval_prompts
        self.eval_idx = -1

        self.resume_epoch = 0
        self.resume_file_idx = 0
        self.resume_batch_idx = 0

        self.generation_service = generation_service
        self.kd_config = kd_config
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        self.logger = Logger('log.txt')

        self.parallel_kwargs, self.data_loader_kwargs, self.sampler_kwargs = self._convert_train_args()
        # initialize a GradScaler. If enabled=False scaler is a no-op
        self._init_scaler()

        # 注意：学习率要根据GPU的数量进行倍增：
        # 在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
        initial_lr = train_config.optim_config.initial_lr

        self.train_model, self.optimizer = self._init_train_model_and_optim(initial_lr)
        self.lr_scheduler = self._init_lr_scheduler(initial_lr, self.optimizer)

        self.criterion, self.kd_loss = self._init_loss()

        self._load_train_model_checkpoint()
        self._apply_restore_ckpt()

        set_seed(default_seed + TrainerTools().parallel.global_rank)

        if TrainerTools().parallel.is_main_process:
            Logger.std_log(f'parallel_kwargs={self.parallel_kwargs}')
            Logger.std_log(f'data_loader_kwargs={self.data_loader_kwargs}')
            Logger.std_log(f'sampler_kwargs={self.sampler_kwargs}')

    def _init_scaler(self):
        device_type = TrainerTools().parallel.device_type
        enable_scaler = TrainerTools().use_amp and (
                    TrainerTools().compute_dtype == 'fp16'
                    or (TrainerTools().compute_dtype == 'auto' and not is_bf16_supported()))
        try:
            self.scaler = torch.amp.GradScaler(device=device_type, enabled=enable_scaler)
        except (AttributeError, TypeError, ValueError):
            if device_type == 'mlu' and hasattr(torch, 'mlu') and hasattr(torch.mlu, 'amp'):
                self.scaler = torch.mlu.amp.GradScaler(enabled=enable_scaler)
            elif device_type == 'npu' and hasattr(torch, 'npu') and hasattr(torch.npu, 'amp'):
                self.scaler = torch.npu.amp.GradScaler(enabled=enable_scaler)
            elif device_type == 'mps' or device_type == 'cpu':
                self.scaler = torch.cuda.amp.GradScaler(enabled=False)
            else:
                self.scaler = torch.cuda.amp.GradScaler(enabled=enable_scaler)

    def _new_model(self, train_config: TrainConfig):
        return LlmModel(train_config.model_config)

    def _init_train_model_and_optim(self, initial_lr: float):
        model = self._new_model(self.train_config)

        if self.train_config.init_state_dict:
            model.load_state_dict(self.train_config.init_state_dict, strict=False)
            self.train_config.init_state_dict = None

        self._check_freeze_llm_model(model)

        if self.train_config.gradient_checkpointing:
            if self.is_ds:
                import deepspeed
                model.gradient_checkpointing_enable(checkpoint_func=deepspeed.checkpointing.checkpoint)
            else:
                model.gradient_checkpointing_enable()

        if TrainerTools().parallel.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            Logger.std_log(f"Total number of parameters: {total_params:,}")

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            Logger.std_log(f"Trainable number of parameters: {trainable_params:,}")

            total_size_bytes = total_params * 4
            total_size_mb = total_size_bytes / (1024 * 1024)
            Logger.std_log(f"Total size of the model: {total_size_mb:.2f} MB")

        model, optim = TrainerTools().parallel.process(
            model=model,
            optimizer=self._config_optim(model, initial_lr),
            kwargs=self.parallel_kwargs
        )

        return model, optim

    def _check_freeze_llm_model(self, model): ...

    def _config_optim(self, model, initial_lr):
        optimizer_cls, use_lion_optim = self._get_optim_cls()

        betas = self.train_config.optim_config.betas
        weight_decay = self.train_config.optim_config.weight_decay

        if betas is None:
            betas = (0.95, 0.98) if use_lion_optim else (0.9, 0.999)

        if weight_decay is None:
            weight_decay = 0.015 if use_lion_optim else 0.01

        no_decay_name_list = ["bias", "norm.weight"]
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if any(nd in name for nd in no_decay_name_list):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        return optimizer_cls(
            optimizer_grouped_parameters,
            lr=initial_lr,
            betas=betas,
            weight_decay=weight_decay
        )

    def _get_optim_cls(self):
        optimizer = None
        use_lion_optim = self.train_config.optim_config.optim_type == 'lion'

        if (self.train_config.optim_config.auto_optimize_optimizer
                and isinstance(TrainerTools().parallel, DsParallel)
                and self.parallel_kwargs
        ):
            import deepspeed
            if ('zero_optimization' in self.parallel_kwargs
                    and 'offload_optimizer' in self.parallel_kwargs['zero_optimization']
                    and self.parallel_kwargs['zero_optimization']['offload_optimizer']['device'] == 'cpu'):
                if torch.cuda.is_available():
                    if self.train_config.optim_config.optim_type == 'lion':
                        if version.parse(importlib.metadata.version("deepspeed")) >= version.parse('0.17.6'):
                            optimizer = deepspeed.ops.lion.DeepSpeedCPULion
                        else:
                            optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam
                            use_lion_optim = False
                            if TrainerTools().parallel.is_main_process:
                                Logger.std_log(
                                    'When set offload_optimizer, lion optim is unsupported, so set optim to adam!!!!!')
                    else:
                        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam
            else:
                if torch.cuda.is_available():
                    if self.train_config.optim_config.optim_type == 'lion':
                        optimizer = deepspeed.ops.lion.FusedLion
                    else:
                        optimizer = deepspeed.ops.adam.FusedAdam

        if not optimizer:
            if self.train_config.optim_config.optim_type == 'lion':
                try:
                    import lion_pytorch
                except:
                    raise Exception(
                        'lion is not detected, please use `pip3 install lion_pytorch` to install or set optim_type to adam')

                optimizer = lion_pytorch.Lion
            else:
                optimizer = torch.optim.AdamW

        return optimizer, use_lion_optim

    def _init_lr_scheduler(self, initial_lr: float, optimizer) -> LRScheduler:
        if self.train_config.optim_config.enable_lr_scheduler:
            warmup_iters = self.train_config.optim_config.warmup_iters
            min_lr = self.train_config.optim_config.min_lr
            max_lr = self.train_config.optim_config.max_lr
            cosine_annealing_period = self.train_config.optim_config.cosine_annealing_period
            cosine_annealing_period_mul = self.train_config.optim_config.cosine_annealing_period_mul

            return WarmupCosineAnnealingLRScheduler(
                optimizer=optimizer,
                warmup_iters=warmup_iters,
                initial_lr=initial_lr,
                min_lr=min_lr,
                max_lr=max_lr,
                cosine_annealing_period=cosine_annealing_period,
                cosine_annealing_period_mul=cosine_annealing_period_mul,
                need_log=TrainerTools().parallel.is_main_process
            )

        return NoneLRScheduler(initial_lr)

    def _init_loss(self):
        critical_tokens: Optional[List[int]] = None
        critical_alpha: float = 1.0
        if self.train_config.loss_config.critical_tokens:
            critical_tokens = self.train_config.loss_config.critical_tokens
            critical_alpha = self.train_config.loss_config.critical_alpha

        criterion = LMLoss(
            critical_tokens=critical_tokens,
            critical_alpha=critical_alpha,
            vocab_size=TrainerTools().tokenizer.vocab_size
        )

        kd_loss = KDLoss() if self.kd_config else None

        return criterion, kd_loss

    def _load_train_model_checkpoint(self):
        load_checkpoint(
            self.train_model,
            optimizer=self.optimizer,
            device=TrainerTools().parallel.device
        )

    def _apply_restore_ckpt(self):
        steps_dict = load_steps()
        if steps_dict:
            self.resume_epoch = steps_dict.get('epoch', 0)
            self.resume_file_idx = steps_dict.get('file_idx', 0)
            self.resume_batch_idx = steps_dict.get('batch_idx', 0)

            self.lr_scheduler.restore_ckpt_dict(steps_dict)

            if TrainerTools().parallel.is_main_process:
                Logger.std_log(f'restore steps_dict={steps_dict}')

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs: Optional[Dict[str, Any]] = None
        if isinstance(TrainerTools().parallel, DsParallel) and self.train_config.ds_config:
            parallel_kwargs = {
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'gradient_clipping': self.train_config.ds_config.gradient_clipping,
                'train_micro_batch_size_per_gpu': self.train_config.batch_size
            }

            if self.train_config.ds_config.wall_clock_breakdown:
                parallel_kwargs['wall_clock_breakdown'] = True

            if (self.train_config.ds_config.flops_profiler
                    and self.train_config.ds_config.flops_profiler.enabled):
                flops_cfg = self.train_config.ds_config.flops_profiler
                parallel_kwargs['flops_profiler'] = {
                    'enabled': flops_cfg.enabled,
                    'profile_step': flops_cfg.profile_step,
                    'module_depth': flops_cfg.module_depth,
                    'top_modules': flops_cfg.top_modules,
                    'detailed': flops_cfg.detailed,
                    'output_file': flops_cfg.output_file
                }

            if self.train_config.ds_config.zero_config:
                zero_config = self.train_config.ds_config.zero_config
                zero_optimization: Dict[str, Any] = {'stage': zero_config.stage}

                if zero_config.allgather_partitions is not None:
                    zero_optimization['allgather_partitions'] = zero_config.allgather_partitions
                if zero_config.allgather_bucket_size is not None:
                    zero_optimization['allgather_bucket_size'] = zero_config.allgather_bucket_size
                if zero_config.overlap_comm is not None:
                    zero_optimization['overlap_comm'] = zero_config.overlap_comm
                if zero_config.reduce_scatter is not None:
                    zero_optimization['reduce_scatter'] = zero_config.reduce_scatter
                if zero_config.reduce_bucket_size is not None:
                    zero_optimization['reduce_bucket_size'] = zero_config.reduce_bucket_size
                if zero_config.contiguous_gradients is not None:
                    zero_optimization['contiguous_gradients'] = zero_config.contiguous_gradients
                if zero_config.ignore_unused_parameters:
                    zero_optimization['ignore_unused_parameters'] = True
                if zero_config.communication_data_type:
                    zero_optimization['communication_data_type'] = zero_config.communication_data_type

                if isinstance(zero_config, (DsZero2Config, DsZero3Config)):
                    if zero_config.offload_optimizer is not None:
                        zero_optimization['offload_optimizer'] = {
                            "device": zero_config.offload_optimizer.device,
                            "pin_memory": zero_config.offload_optimizer.pin_memory
                        }

                        if zero_config.offload_optimizer.device == 'nvme':
                            if zero_config.offload_optimizer.nvme_path: zero_optimization['offload_optimizer']["nvme_path"] = zero_config.offload_optimizer.nvme_path
                            if zero_config.offload_optimizer.buffer_count: zero_optimization['offload_optimizer']["buffer_count"] = zero_config.offload_optimizer.buffer_count
                            if zero_config.offload_optimizer.buffer_size: zero_optimization['offload_optimizer']["buffer_size"] = zero_config.offload_optimizer.buffer_size
                            if zero_config.offload_optimizer.max_in_cpu: zero_optimization['offload_optimizer']["max_in_cpu"] = zero_config.offload_optimizer.max_in_cpu
                    if zero_config.offload_param is not None:
                        zero_optimization['offload_param'] = {
                            "device": zero_config.offload_param.device,
                            "pin_memory": zero_config.offload_param.pin_memory
                        }

                        if zero_config.offload_param.device == 'nvme':
                            if zero_config.offload_param.nvme_path: zero_optimization['offload_param']["nvme_path"] = zero_config.offload_param.nvme_path
                            if zero_config.offload_param.buffer_count: zero_optimization['offload_param']["buffer_count"] = zero_config.offload_param.buffer_count
                            if zero_config.offload_param.buffer_size: zero_optimization['offload_param']["buffer_size"] = zero_config.offload_param.buffer_size
                            if zero_config.offload_param.max_in_cpu: zero_optimization['offload_param']["max_in_cpu"] = zero_config.offload_param.max_in_cpu

                if isinstance(zero_config, DsZero3Config):
                    if zero_config.sub_group_size is not None:
                        zero_optimization['sub_group_size'] = zero_config.sub_group_size
                    if zero_config.stage3_prefetch_bucket_size is not None:
                        zero_optimization['stage3_prefetch_bucket_size'] = zero_config.stage3_prefetch_bucket_size
                    if zero_config.stage3_param_persistence_threshold is not None:
                        zero_optimization['stage3_param_persistence_threshold'] = zero_config.stage3_param_persistence_threshold
                    if zero_config.stage3_max_live_parameters is not None:
                        zero_optimization['stage3_max_live_parameters'] = zero_config.stage3_max_live_parameters
                    if zero_config.stage3_max_reuse_distance is not None:
                        zero_optimization['stage3_max_reuse_distance'] = zero_config.stage3_max_reuse_distance
                    if zero_config.stage3_gather_16bit_weights_on_model_save is not None:
                        zero_optimization['stage3_gather_16bit_weights_on_model_save'] = zero_config.stage3_gather_16bit_weights_on_model_save
                    if zero_config.memory_efficient_linear is not None:
                        zero_optimization['memory_efficient_linear'] = zero_config.memory_efficient_linear
                    if zero_config.zero_quantized_weights:
                        zero_optimization['zero_quantized_weights'] = True
                    if zero_config.zero_hpz_partition_size > 1:
                        zero_optimization['zero_hpz_partition_size'] = zero_config.zero_hpz_partition_size
                    if zero_config.zero_quantized_gradients:
                        zero_optimization['zero_quantized_gradients'] = True

                parallel_kwargs['zero_optimization'] = zero_optimization

            compute_dtype = TrainerTools().compute_dtype
            enable_bf16 = False
            enable_fp16 = False

            if compute_dtype == 'bf16':
                enable_bf16 = True
            elif compute_dtype == 'fp16':
                enable_fp16 = True
            elif compute_dtype == 'fp32':
                pass
            elif (self.train_config.ds_config.bf16_config is not None
                    and self.train_config.ds_config.bf16_config.enabled
                    and is_bf16_supported()):
                enable_bf16 = True
            elif self.train_config.ds_config.fp16_config and is_fp16_supported():
                enable_fp16 = True

            if enable_bf16:
                bf16: Dict[str, Any] = {'enabled': True}
                parallel_kwargs['bf16'] = bf16
            elif enable_fp16:
                fp16: Dict[str, Any] = {'enabled': True}
                fp16_config = self.train_config.ds_config.fp16_config
                if fp16_config:
                    fp16.update({
                        'loss_scale': fp16_config.loss_scale,
                        'loss_scale_window': fp16_config.loss_scale_window,
                        'initial_scale_power': fp16_config.initial_scale_power,
                        'hysteresis': fp16_config.hysteresis,
                        'min_loss_scale': fp16_config.min_loss_scale
                    })

                    if fp16_config.fp16_opt_level is not None:
                        fp16['fp16_opt_level'] = fp16_config.fp16_opt_level

                parallel_kwargs['fp16'] = fp16

            if self.train_config.ds_config.activation_checkpointing:
                activation_checkpointing_config = self.train_config.ds_config.activation_checkpointing
                activation_checkpointing: Dict[str, Any] = {
                    'partition_activations': activation_checkpointing_config.partition_activations,
                    'cpu_checkpointing': activation_checkpointing_config.cpu_checkpointing,
                    'contiguous_memory_optimization': activation_checkpointing_config.contiguous_memory_optimization,
                    'synchronize_checkpoint_boundary': activation_checkpointing_config.synchronize_checkpoint_boundary,
                    'profile': activation_checkpointing_config.profile
                }

                if activation_checkpointing_config.number_checkpoints is not None:
                    activation_checkpointing['number_checkpoints'] = activation_checkpointing_config.number_checkpoints

                parallel_kwargs['activation_checkpointing'] = activation_checkpointing

        dataloader_args = self.train_config.data_loader_config
        data_loader_kwargs = {
            "batch_size": self.train_config.batch_size,
            "pin_memory": dataloader_args.pin_memory,
            "num_workers": dataloader_args.num_workers,
            "shuffle": dataloader_args.shuffle,
            "drop_last": True,
        }
        sampler_kwargs = {
            "shuffle": dataloader_args.shuffle,
            "drop_last": True,
        }

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _init_ref_model_args(self, model_config: Optional[ModelConfig] = None) -> dict:
        parallel_kwargs = copy.deepcopy(self.parallel_kwargs) if self.parallel_kwargs else None

        if parallel_kwargs and isinstance(TrainerTools().parallel, DsParallel):
            stage = parallel_kwargs.get("zero_optimization", {}).get("stage", 0)
            if model_config is not None:
                hidden_size = model_config.hidden_size
                if hidden_size is not None and stage == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
                    # @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error
                    zero_optimization = parallel_kwargs.get("zero_optimization", {})
                    zero_optimization.update(
                        {
                            "reduce_bucket_size": int(hidden_size * hidden_size),
                            "stage3_param_persistence_threshold": int(10 * hidden_size),
                            "stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
                        }
                    )

            parallel_kwargs.pop('activation_checkpointing', None)
            parallel_kwargs.pop('gradient_clipping', None)
            parallel_kwargs.get("zero_optimization", {}).pop("offload_optimizer", None)

            if stage != 3:
                parallel_kwargs["zero_optimization"] = {"stage": 0}

        return parallel_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]: ...

    def _calc_loss(self, inputs, attention_mask, logits, labels):
        # calc loss
        ce_loss = self.criterion(logits, labels)
        if not self.kd_loss or self.kd_config.kd_coef == 0.0:
            # 不用计算kd_loss
            return ce_loss, ce_loss

        teacher_logits = self.kd_config.teacher_logits_provider(inputs, attention_mask)
        loss = self.kd_loss(logits, teacher_logits, labels)

        return (1 - self.kd_config.kd_coef) * ce_loss + self.kd_config.kd_coef * loss, ce_loss

    def _backward_loss(self, total_loss_unscaled, gradient_accumulation_steps, step = True):
        if isinstance(TrainerTools().parallel, DsParallel):
            self.train_model.backward(total_loss_unscaled)
            if step:
                self.train_model.step()
        else:
            total_loss_scaled = total_loss_unscaled / gradient_accumulation_steps
            self.scaler.scale(total_loss_scaled).backward()

    def _apply_grad_clipping(self):
        if not isinstance(TrainerTools().parallel, DsParallel) and self.lr_scheduler.can_clip_grad():
            self.scaler.unscale_(self.optimizer)

            trainable_params = filter(lambda p: p.requires_grad, self.train_model.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

    def _apply_step(self):
        if not isinstance(TrainerTools().parallel, DsParallel):
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

    def _need_update_step(self, batches_accumulated, is_last_step=False):
        if self.is_ds:
            return self.train_model.is_gradient_accumulation_boundary()

        if self.gradient_accumulation_steps > 1:
            return (batches_accumulated + 1) % self.gradient_accumulation_steps == 0 or is_last_step

        return True

    def _update_step(self):
        self._apply_grad_clipping()
        overflow = False

        if self.is_ds:
            self._apply_step()
            if hasattr(self.train_model, 'optimizer') and hasattr(self.train_model.optimizer, 'overflow'):
                overflow = self.train_model.optimizer.overflow
        else:
            scale_before = self.scaler.get_scale()
            self._apply_step()
            scale_after = self.scaler.get_scale()
            overflow = scale_after < scale_before

        if not overflow:
            self.lr_scheduler.step()

        TrainerTools().parallel.synchronize()

    def _get_eval_data(self) -> Optional[str]:
        if len(self.eval_prompts) == 0:
            return None

        self.eval_idx += 1
        if self.eval_idx == len(self.eval_prompts):
            self.eval_idx = 0

        return self.eval_prompts[self.eval_idx]

    def _get_eval_pixel_values_and_tokens_count(self, eval_idx):
        return None, None

    def _log(self, keys: Dict[str, Any], values: Dict[str, Any]):
        """
        格式：keys_key1: keys_value1, keys_key2: keys_value2 -> values_key1: values_value1, values_key2: values_value2
        """
        if TrainerTools().parallel.is_main_process:
            log_tags = ', '.join([f'{k}: {v}' for k, v in keys.items()])
            log_values = ', '.join([f'{k}: {v}' for k, v in values.items()])

            log_msg = f'{log_tags} -> {log_values}'
            self.logger.log(log_msg)

    def _on_exception(
            self,
            e: Exception,
            epoch: int,
            batch: int
    ):
        exception_file = e.__traceback__.tb_frame.f_globals["__file__"]
        exception_line = e.__traceback__.tb_lineno
        log_msg = f"epoch: {epoch}, batch: {batch} -> {e} at {exception_file} line {exception_line}"
        Logger('exception.txt').log(log_msg, log_to_console=False).release()

        raise e

    def _eval(self, tag: str):
        eval_prompt = self._get_eval_data()
        if eval_prompt is None:
            return

        eval_pixel_values, tokens_per_image = self._get_eval_pixel_values_and_tokens_count(self.eval_idx)
        if tokens_per_image is None:
            tokens_per_image = -1
            eval_pixel_values = None

        if self.generation_service is not None:
            tokens = TrainerTools().tokenizer.encode(eval_prompt, unsqueeze=True, covert_tensor=True)
            response_ids = self.generation_service(
                self.train_model, tokens, self.train_config.eval_config,
                'eval', eval_pixel_values, tokens_per_image
            )

            if TrainerTools().parallel.is_main_process and response_ids:
                gen_text = TrainerTools().tokenizer.decode(response_ids[0])
                with open(os.path.join(_get_log_dir(), 'gen.txt'), 'a') as f:
                    f.write(f"{tag}, gen->{eval_prompt}{gen_text}\n")
        else:
            with unwrap_model_for_generation(self.train_model) as eval_model:
                if TrainerTools().parallel.is_main_process:
                    eval_model = self._check_eval_model(eval_model)
                    eval_model.eval()

                    tokens = TrainerTools().tokenizer.encode(eval_prompt, unsqueeze=True, covert_tensor=True)
                    max_new_tokens = max(self.train_config.eval_config.max_seq_len - tokens.shape[1], 0)

                    gen_result = generate(
                        eval_model,
                        prompt=tokens,
                        max_new_tokens=max_new_tokens,
                        temperature=self.train_config.eval_config.temperature,
                        top_k=self.train_config.eval_config.top_k,
                        top_p=self.train_config.eval_config.top_p,
                        repetition_penalty=self.train_config.eval_config.repetition_penalty,
                        exclude_penalty_tokens=self.train_config.eval_config.exclude_penalty_tokens,
                        suppress_tokens=self.train_config.eval_config.suppress_tokens,
                        pixel_values=eval_pixel_values,
                        tokens_per_image=tokens_per_image,
                        device=TrainerTools().parallel.device
                    )

                    with open(os.path.join(_get_log_dir(), 'gen.txt'), 'a') as f:
                        f.write(f"{tag}, gen->{gen_result}\n")

                    eval_model.train()

        TrainerTools().parallel.wait('eval')

    def _check_eval_model(self, eval_model):
        return eval_model

    def _on_batch_end(self, tag: str):
        self._eval(f'sign:batch/{tag}')

    def _on_epoch_end(self, tag: str):
        self._eval(f'sign:epoch/{tag}')

    def _on_file_start(
            self,
            epoch: int,
            file_name: str
    ):
        if TrainerTools().parallel.is_main_process:
            self.logger.log(f"====epoch: {epoch}, start train {file_name}====", log_to_console=False)

    def _avg_loss(
            self,
            losses: List[float],
            batches_accumulated
    ) -> List[float]:
        loss_tensors =[
            torch.tensor(loss / batches_accumulated, device=TrainerTools().parallel.device)
            for loss in losses
        ]

        stacked_losses = torch.stack(loss_tensors)
        # 跨卡同步平均
        if TrainerTools().parallel.parallel_train:
            if TrainerTools().parallel.device_type == 'mlu':
                dist.all_reduce(stacked_losses, op=dist.ReduceOp.SUM)
                stacked_losses.div_(TrainerTools().parallel.world_size)
            else:
                dist.all_reduce(stacked_losses, dist.ReduceOp.AVG)

        return stacked_losses.detach().cpu().tolist()

    def _get_pixel_values(self, batch_data):
        return None

    def _calc_attention_mask(self, inputs):
        return inputs != TrainerTools().tokenizer.pad

    def train(self):
        # 梯度累积步数
        loss_accumulation = 0.0
        aux_loss_accumulation = 0.0
        ce_loss_accumulation = 0.0
        batches_accumulated = 0

        for epoch in range(self.resume_epoch, self.train_config.n_epochs):
            self.train_model.train()
            file_count = len(self.train_config.file_dataset)
            start_file_idx = self.resume_file_idx if epoch == self.resume_epoch else 0

            for file_idx in range(start_file_idx, file_count):
                dataset, file_path = self._create_dataset(file_idx)
                train_data_loader = TrainerTools().parallel.process_dataloader(
                    dataset=dataset,
                    data_loader_kwargs=self.data_loader_kwargs,
                    sampler_kwargs=self.sampler_kwargs
                )

                last_ckpt_batch = 0
                batch_count_per_file = len(train_data_loader)

                TrainerTools().parallel.on_epoch_start(epoch)
                self._on_file_start(epoch, file_path)

                skip_batches = 0
                if epoch == self.resume_epoch and file_idx == self.resume_file_idx:
                    skip_batches = self.resume_batch_idx
                    if skip_batches > 0 and TrainerTools().parallel.is_main_process:
                        Logger.std_log(f"Fast forwarding {skip_batches} batches in {file_path}...")

                data_iterator = iter(train_data_loader)

                if skip_batches > 0:
                    data_iterator = islice(data_iterator, skip_batches, None)
                    last_ckpt_batch = skip_batches

                for batch, batch_data in enumerate(data_iterator):
                    batch = skip_batches + batch

                    inputs = batch_data['inputs']
                    labels = batch_data['labels']

                    try:
                        inputs, labels = inputs.to(TrainerTools().parallel.device), labels.to(TrainerTools().parallel.device)
                        attention_mask = self._calc_attention_mask(inputs)
                        pixel_values = self._get_pixel_values(batch_data)

                        with autocast(TrainerTools().parallel.device_type):
                            result = self.train_model(
                                inputs,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values
                            )

                            # calc loss
                            loss, ce_loss = self._calc_loss(inputs, attention_mask, result['logits'], labels)
                            if result['aux_loss'] is not None:
                                aux_loss = result['aux_loss'].to(loss.dtype)
                            else:
                                aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                        total_loss_unscaled = loss + aux_loss

                        is_last_step = (
                            epoch == self.train_config.n_epochs - 1
                            and file_idx == file_count - 1
                            and batch == batch_count_per_file - 1
                        )

                        need_update_step = self._need_update_step(batches_accumulated, is_last_step)
                        self._backward_loss(total_loss_unscaled, self.gradient_accumulation_steps)

                        loss_accumulation += total_loss_unscaled.detach().item()
                        aux_loss_accumulation += aux_loss.detach().item()
                        ce_loss_accumulation += ce_loss.detach().item()
                        batches_accumulated += 1

                        if need_update_step:
                            self._update_step()

                            avg_loss, avg_aux_loss, avg_ce_loss = self._avg_loss(
                                losses=[
                                    loss_accumulation,
                                    aux_loss_accumulation,
                                    ce_loss_accumulation
                                ],
                                batches_accumulated=batches_accumulated
                            )

                            try:
                                perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 20 else float('inf')
                            except OverflowError:
                                perplexity = float('inf')

                            self._log(
                                keys={
                                    'epoch': epoch,
                                    'file': f'{file_idx + 1}/{file_count}',
                                    'batch': f'{batch + 1}/{batch_count_per_file}'
                                },
                                values={
                                    'loss/total': avg_loss,
                                    'loss/moe_aux': avg_aux_loss,
                                    'metrics/ppl': round(perplexity, 4) if avg_ce_loss > 0 else float('inf')
                                }
                            )

                            # reset to default
                            loss_accumulation = 0.0
                            aux_loss_accumulation = 0.0
                            ce_loss_accumulation = 0.0
                            batches_accumulated = 0

                            if (batch - last_ckpt_batch) >= self.train_config.save_and_eval_interval:
                                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                                save_steps(
                                    epoch=epoch,
                                    file_idx=file_idx,
                                    batch_idx=batch + 1,
                                    lr_scheduler=self.lr_scheduler
                                )

                                last_ckpt_batch = batch
                                self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')
                    except Exception as e:
                        self._on_exception(e, epoch, batch)

                try:
                    # 一个文件训练结束后，清理内存
                    del train_data_loader
                    del dataset
                    del data_iterator
                    del batch_data
                    del inputs
                    del labels
                    del attention_mask
                    del result
                    del loss
                    del total_loss_unscaled
                    del aux_loss
                    del pixel_values
                except UnboundLocalError: ...

                if hasattr(TrainerTools().parallel, '_sampler'):
                    TrainerTools().parallel._sampler = None

                gc.collect()
                empty_cache()

            # end epoch

            # reset resume state
            self.resume_file_idx = 0
            self.resume_batch_idx = 0

            save_checkpoint(model=self.train_model, optimizer=self.optimizer)
            save_steps(
                epoch=epoch + 1,
                file_idx=0,
                batch_idx=0,
                lr_scheduler=self.lr_scheduler
            )

            TrainerTools().parallel.on_epoch_end(epoch)
            self._on_epoch_end(tag=f'epoch:{epoch}')

        TrainerTools().parallel.destroy()
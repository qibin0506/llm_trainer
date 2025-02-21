import time
from contextlib import nullcontext
from typing import Optional, Tuple, List

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset
from llama import LlamaModel

from .parallel_ds import DsParallel
from .parallel_fsdp import FsdpParallel
from .tools import TrainerTools
from .loss import LMLoss, KDLoss
from .log import log
from .dataset import TextDataset

from .train_configs import (
    TrainConfig,
    DsZero2Config,
    DsZero3Config
)

from .scheduler import (
    LRScheduler,
    CosineAnnealingWarmupLRScheduler,
    NoneLRScheduler
)

from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
    load_steps,
    save_steps,
)
from .utils import (
    set_seed,
    pretrain_collate_fn,
)

from .trainer_log import (
    on_batch_end,
    on_exception,
    on_epoch_end,
    on_file_start,
    log_loss
)


class Trainer:
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            prompt_on_batch: str,
            prompt_on_epoch: str,
    ):
        set_seed()

        self.train_config: TrainConfig = train_config
        self.prompt_on_batch: str = prompt_on_batch
        self.prompt_on_epoch: str = prompt_on_epoch

        parallel_kwargs, data_loader_kwargs, sampler_kwargs = self._convert_train_args()
        self.data_loader_kwargs: dict[str, any] = data_loader_kwargs
        self.sampler_kwargs: dict[str, any] = sampler_kwargs

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scalar = torch.GradScaler(enabled=TrainerTools().use_amp)

        batch_count = train_config.all_data_size // TrainerTools().parallel.world_size // train_config.batch_size

        log(f"real batch count: {batch_count}")

        if train_config.gradient_accumulation_steps > 1:
            batch_count = batch_count // train_config.gradient_accumulation_steps

        # 学习率要根据GPU的数量进行倍增：
        # 在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
        initial_lr = train_config.lr_scheduler_config.initial_lr * TrainerTools().parallel.world_size

        self.train_model, self.optimizer = self._init_train_model_and_optim(initial_lr, parallel_kwargs)
        self.lr_scheduler = self._init_lr_scheduler(batch_count, initial_lr)
        self.eval_model: Optional[nn.Module] = self._init_eval_model()

        self.criterion, self.kd_loss = self._init_loss()

        self.ctx = torch.autocast(
            device_type=TrainerTools().parallel.device_type,
            dtype=TrainerTools().dtype,
            enabled=TrainerTools().use_amp,
            # fsdp模式，需要将cache_enabled设置为false
            # https://www.zhihu.com/question/642793891
            cache_enabled=False if isinstance(self.train_model, FSDP) else None
        ) if TrainerTools().use_amp else nullcontext()

        load_checkpoint(
            self.train_model,
            optimizer=self.optimizer,
            device=TrainerTools().parallel.device
        )

        last_global_steps, last_lr_steps = load_steps(0, -1)
        self.last_global_steps = last_global_steps
        log(f'last_global_steps={last_global_steps}, last_lr_steps={last_lr_steps}')

        if last_lr_steps != -1:
            self.lr_scheduler.update_steps(last_lr_steps)

    def _init_train_model_and_optim(
            self,
            initial_lr: float,
            parallel_kwargs
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        model = LlamaModel(self.train_config.llama_config)
        if TrainerTools().parallel.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            log(f"Total number of parameters: {total_params:,}")

            total_size_bytes = total_params * 4
            total_size_mb = total_size_bytes / (1024 * 1024)
            log(f"Total size of the model: {total_size_mb:.2f} MB")

        optim = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.1)
        model, optim = TrainerTools().parallel.process(
            model=model,
            optimizer=optim,
            kwargs=parallel_kwargs
        )

        return model, optim

    def _init_eval_model(self) -> Optional[nn.Module]:
        if TrainerTools().parallel.is_main_process:
            return LlamaModel(self.train_config.llama_config).to('cpu')

        return None

    def _init_lr_scheduler(self, batch_count: int, initial_lr: float) -> LRScheduler:
        if self.train_config.lr_scheduler_config.enable_lr_scheduler:
            train_iters = batch_count * self.train_config.n_epochs
            warmup_iters = int(self.train_config.lr_scheduler_config.warmup_iters_ratio * train_iters)
            min_lr = self.train_config.lr_scheduler_config.min_lr_ratio * initial_lr
            max_lr = self.train_config.lr_scheduler_config.max_lr * TrainerTools().parallel.world_size

            return CosineAnnealingWarmupLRScheduler(
                optimizer=self.optimizer,
                warmup_iters=warmup_iters,
                initial_lr=initial_lr,
                min_lr=min_lr,
                max_lr=max_lr,
                total_iters=train_iters
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

        kd_loss = KDLoss() if self.train_config.kd_config else None

        return criterion, kd_loss

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        if isinstance(TrainerTools().parallel, DsParallel) and self.train_config.ds_config:
            parallel_kwargs = {
                'gradient_accumulation_steps': 1,
                'gradient_clipping': self.train_config.ds_config.gradient_clipping,
                'train_micro_batch_size_per_gpu': self.train_config.batch_size
            }

            if self.train_config.ds_config.zero_config:
                zero_config = self.train_config.ds_config.zero_config
                zero_optimization = {'stage': zero_config.stage}

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

                if isinstance(zero_config, DsZero2Config) or isinstance(zero_config, DsZero3Config):
                    if zero_config.offload_optimizer is not None:
                        zero_optimization['offload_optimizer'] = zero_config.offload_optimizer
                    if zero_config.offload_param is not None:
                        zero_optimization['offload_param'] = zero_config.offload_param

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

                parallel_kwargs['zero_optimization'] = zero_optimization

            if self.train_config.ds_config.fp16_config:
                fb16_config = self.train_config.ds_config.fp16_config
                fp16 = {
                    'enabled': fb16_config.enabled,
                    'loss_scale': fb16_config.loss_scale,
                    'loss_scale_window': fb16_config.loss_scale_window,
                    'initial_scale_power': fb16_config.initial_scale_power,
                    'hysteresis': fb16_config.hysteresis,
                    'min_loss_scale': fb16_config.min_loss_scale
                }

                if fb16_config.fp16_opt_level is not None:
                    fp16['fp16_opt_level'] = fb16_config.fp16_opt_level

                parallel_kwargs['fp16'] = fp16

            if self.train_config.ds_config.bf16_config:
                bf16_config = self.train_config.ds_config.bf16_config
                bf16 = {
                    'enabled': bf16_config.enabled
                }
                parallel_kwargs['bf16'] = bf16

            if self.train_config.ds_config.activation_checkpointing:
                activation_checkpointing_config = self.train_config.ds_config.activation_checkpointing
                activation_checkpointing = {
                    'partition_activations': activation_checkpointing_config.partition_activations,
                    'cpu_checkpointing': activation_checkpointing_config.cpu_checkpointing,
                    'contiguous_memory_optimization': activation_checkpointing_config.contiguous_memory_optimization,
                    'synchronize_checkpoint_boundary': activation_checkpointing_config.synchronize_checkpoint_boundary,
                    'profile': activation_checkpointing_config.profile
                }

                if activation_checkpointing_config.number_checkpoints is not None:
                    activation_checkpointing['number_checkpoints'] = activation_checkpointing_config.number_checkpoints

                parallel_kwargs['activation_checkpointing'] = activation_checkpointing
        elif isinstance(TrainerTools().parallel, FsdpParallel) and self.train_config.fsdp_config:
            parallel_kwargs = {
                'transformer_layer_cls': self.train_config.fsdp_config.transformer_layer_cls,
                'wrap_policy_num_params': self.train_config.fsdp_config.wrap_policy_num_params,
                'cpu_offload': self.train_config.fsdp_config.cpu_offload,
                'offload_params': self.train_config.fsdp_config.offload_params
            }
        else:
            parallel_kwargs = None

        dataloader_args = self.train_config.data_loader_config
        data_loader_kwargs = {
            "batch_size": self.train_config.batch_size,
            "pin_memory": dataloader_args.data_loader_pin_memory,
            "collate_fn": pretrain_collate_fn,
            "num_workers": dataloader_args.data_loader_num_workers,
            "shuffle": dataloader_args.data_loader_shuffle,
            "drop_last": dataloader_args.data_loader_drop_last,
        }
        sampler_kwargs = {
            "shuffle": dataloader_args.data_loader_shuffle,
            "drop_last": dataloader_args.data_loader_drop_last,
        }

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_path) -> Dataset:
        max_position_embeddings = self.train_config.llama_config.max_position_embeddings
        return TextDataset(file_path, max_position_embeddings, max_position_embeddings)

    def _calc_loss(self, inputs, attention_mask, logits, labels):
        # calc loss
        loss = self.criterion(logits, labels)

        # 知识蒸馏loss
        if self.kd_loss:
            teacher_logits = self.train_config.kd_config.teacher_logits_provider(inputs, attention_mask)
            distil_loss = self.kd_loss(logits, teacher_logits, labels)
            loss = (1 - self.train_config.kd_config.kd_coef) * loss + self.train_config.kd_config.kd_coef * distil_loss

        return loss

    def _backward_loss(self, loss):
        if isinstance(TrainerTools().parallel, DsParallel):
            self.train_model.backward(loss)
        else:
            self.scalar.scale(loss).backward()

    def _step(self):
        self.lr_scheduler.step()
        if isinstance(TrainerTools().parallel, DsParallel):
            self.train_model.step()
        else:
            self.scalar.step(self.optimizer)
            # optimizer.step()
            self.scalar.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)

        TrainerTools().parallel.synchronize()

    def _log_loss(
            self,
            epoch: int,
            file_idx: int,
            file_count: int,
            batch: int,
            batch_count: int,
            loss
    ):
        log_loss(epoch, file_idx, file_count, batch, batch_count, loss, self.lr_scheduler.cur_lr)

    def _on_exception(self, e: Exception, epoch: int, batch: int):
        on_exception(e, epoch, batch)

    def _on_batch_end(
            self,
            epoch: int,
            batch: int,
    ):
        on_batch_end(
            self.eval_model,
            epoch,
            batch,
            self.prompt_on_batch,
            self.train_config.llama_config.max_position_embeddings
        )

    def _on_epoch_end(self, epoch: int):
        on_epoch_end(
            self.eval_model,
            epoch,
            self.prompt_on_epoch,
            self.train_config.llama_config.max_position_embeddings
        )

    def train(self):
        # 梯度累积步数
        gradient_accumulation_steps = self.train_config.gradient_accumulation_steps
        global_steps = 0
        loss_accumulation = 0.0
        skipping_train = False

        for epoch in range(self.train_config.n_epochs):
            self.train_model.train()
            file_count = len(self.train_config.all_files)

            for file_idx in range(file_count):
                file_path = self.train_config.all_files[file_idx]

                dataset = self._create_dataset(file_path)
                train_data_loader = TrainerTools().parallel.process_dataloader(
                    dataset=dataset,
                    data_loader_kwargs=self.data_loader_kwargs,
                    sampler_kwargs=self.sampler_kwargs
                )

                last_ckpt_batch = 0
                batch_count_per_file = len(train_data_loader)

                TrainerTools().parallel.on_epoch_start(epoch)
                on_file_start(epoch, file_path)

                for batch, (inputs, labels) in enumerate(train_data_loader):
                    global_steps += 1
                    if global_steps < self.last_global_steps:
                        skipping_train = True
                        continue

                    skipping_train = False

                    # 是否需要更新梯度
                    if gradient_accumulation_steps > 1:
                        need_update_grad = (batch + 1) % gradient_accumulation_steps == 0 or batch == batch_count_per_file - 1
                    else:
                        need_update_grad = True

                    try:
                        inputs, labels = inputs.to(TrainerTools().parallel.device), labels.to(TrainerTools().parallel.device)
                        attention_mask = inputs != TrainerTools().tokenizer.pad

                        if TrainerTools().parallel.parallel_train:
                            # in DDP training we only need to sync gradients at the last micro step.
                            # the official way to do this is with model.no_sync() context manager, but
                            # I really dislike that this bloats the code and forces us to repeat code
                            # looking at the source of that context manager, it just toggles this variable
                            self.train_model.require_backward_grad_sync = need_update_grad

                        with self.ctx:
                            logits, _ = self.train_model(inputs, attention_mask=attention_mask)
                            # calc loss
                            loss = self._calc_loss(inputs, attention_mask, logits, labels)

                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                        loss_accumulation += loss.detach()
                        self._backward_loss(loss)

                        if need_update_grad:
                            # todo check all_reduce??
                            if TrainerTools().parallel.parallel_train:
                                dist.all_reduce(loss_accumulation, dist.ReduceOp.AVG)

                            # ds模式已经集成gradient_clipping
                            if not isinstance(TrainerTools().parallel, DsParallel) and self.lr_scheduler.can_clip_grad():
                                # clip grad
                                self.scalar.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)

                            self._step()
                            self._log_loss(epoch, file_idx, file_count, batch, batch_count_per_file, loss_accumulation.item())
                            # reset to default
                            loss_accumulation = 0.0
                    except Exception as e:
                        self._on_exception(e, epoch, batch)
                    finally:
                        if need_update_grad and (batch - last_ckpt_batch) >= self.train_config.eval_batch_interval:
                            save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                            save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                            last_ckpt_batch = batch
                            self._on_batch_end(epoch, batch)

                        try:
                            del loss
                        except UnboundLocalError:
                            pass

            # end epoch
            if not skipping_train:
                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                TrainerTools().parallel.on_epoch_end(epoch)
                self._on_epoch_end(epoch)

        # 等待checkpoint保存完成
        time.sleep(10)
        TrainerTools().parallel.destroy()

"""
todo: 
0. 实现按照token数据确定batch大小，不固定batch的方案 done
1. 处理异常重启
2. Yarn和phi3的Phi3LongRoPEScaledRotaryEmbedding调研
3. MLA调研
4. DPO调研
5. inference使用缓存model，每个进程缓存一个
6. 多模态
"""
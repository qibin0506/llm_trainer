from contextlib import nullcontext
from typing import Optional, Tuple

import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import Dataset
from llama import LlamaModel

from .train_args import TrainArgs
from .parallel_fsdp import FsdpParallel
from .tools import TrainerTools
from .loss import LMLoss, KDLoss
from .log import log
from .dataset import TextDataset

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
)


class Trainer:
    def __init__(
            self,
            *,
            train_args: TrainArgs,
            prompt_on_batch: str,
            prompt_on_epoch: str,
    ):
        set_seed()

        self.train_args: TrainArgs = train_args
        self.prompt_on_batch: str = prompt_on_batch
        self.prompt_on_epoch: str = prompt_on_epoch

        parallel_kwargs, data_loader_kwargs, sampler_kwargs = self._convert_train_args()
        self.data_loader_kwargs: dict[str, any] = data_loader_kwargs
        self.sampler_kwargs: dict[str, any] = sampler_kwargs

        self.train_model: nn.Module = self._init_train_model(parallel_kwargs)
        self.eval_model: Optional[nn.Module] = self._init_eval_model()

        self.criterion = LMLoss()
        self.kd_loss = KDLoss() if train_args.kd_args else None

        self.ctx = torch.autocast(
            device_type=TrainerTools().parallel.device_type,
            dtype=TrainerTools().dtype,
            enabled=TrainerTools().use_amp
        ) if TrainerTools().use_amp else nullcontext()

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scalar = torch.GradScaler(enabled=TrainerTools().use_amp)

        # 梯度累积步数
        batch_count = train_args.all_data_size // TrainerTools().parallel.world_size // train_args.batch_size

        log(f"real batch count: {batch_count}")

        if self.train_args.gradient_accumulation_steps > 1:
            batch_count = batch_count // self.train_args.gradient_accumulation_steps

        # 学习率要根据GPU的数量进行倍增：
        # 在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
        initial_lr = train_args.lr_scheduler_args.initial_lr * TrainerTools().parallel.world_size
        self.optimizer = self._init_optimizer(initial_lr)
        self.lr_scheduler = self._init_lr_scheduler(batch_count, initial_lr)

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

    def _init_train_model(self, parallel_kwargs) -> nn.Module:
        model = TrainerTools().parallel.process_model(
            LlamaModel(self.train_args.llama_config),
            kwargs=parallel_kwargs
        )

        if TrainerTools().parallel.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            log(f"Total number of parameters: {total_params:,}")

            total_size_bytes = total_params * 4
            total_size_mb = total_size_bytes / (1024 * 1024)
            log(f"Total size of the model: {total_size_mb:.2f} MB")

        return model

    def _init_eval_model(self) -> Optional[nn.Module]:
        if TrainerTools().parallel.is_main_process:
            return LlamaModel(self.train_args.llama_config).to('cpu')

        return None

    def _init_optimizer(self, initial_lr: float) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.train_model.parameters(), lr=initial_lr, weight_decay=0.1)

    def _init_lr_scheduler(self, batch_count: int, initial_lr: float) -> LRScheduler:
        if self.train_args.lr_scheduler_args.enable_lr_scheduler:
            train_iters = batch_count * self.train_args.n_epochs
            warmup_iters = int(self.train_args.lr_scheduler_args.warmup_iters_ratio * train_iters)
            min_lr = self.train_args.lr_scheduler_args.min_lr_ratio * initial_lr
            max_lr = self.train_args.lr_scheduler_args.max_lr * TrainerTools().parallel.world_size

            return CosineAnnealingWarmupLRScheduler(
                optimizer=self.optimizer,
                warmup_iters=warmup_iters,
                initial_lr=initial_lr,
                min_lr=min_lr,
                max_lr=max_lr,
                total_iters=train_iters
            )

        return NoneLRScheduler()

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        if isinstance(TrainerTools().parallel, FsdpParallel) and self.train_args.fsdp_args:
            parallel_kwargs = {
                'transformer_layer_cls': self.train_args.fsdp_args.transformer_layer_cls,
                'wrap_policy_num_params': self.train_args.fsdp_args.wrap_policy_num_params,
                'cpu_offload': self.train_args.fsdp_args.cpu_offload,
                'offload_params': self.train_args.fsdp_args.offload_params
            }
        else:
            parallel_kwargs = None

        dataloader_args = self.train_args.data_loader_args
        data_loader_kwargs = {
            "batch_size": self.train_args.batch_size,
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
        max_position_embeddings = self.train_args.llama_config.max_position_embeddings
        return TextDataset(file_path, max_position_embeddings, max_position_embeddings)

    def _calc_loss(self, inputs, attention_mask, logits, labels):
        # calc loss
        loss = self.criterion(logits, labels)

        # 知识蒸馏loss
        if self.kd_loss:
            teacher_logits = self.train_args.kd_args.teacher_logits_provider(inputs, attention_mask)
            distil_loss = self.kd_loss(logits, teacher_logits, labels)
            loss = (1 - self.train_args.kd_args.kd_coef) * loss + self.train_args.kd_args.kd_coef * distil_loss

        return loss

    def _on_exception(self, e: Exception, epoch: int, batch: int):
        on_exception(e, epoch, batch)

    def _on_batch_end(
            self,
            epoch: int,
            batch: int,
            batch_count_per_file: int,
            loss,
            need_update_grad: bool
    ):
        on_batch_end(
            self.eval_model,
            epoch,
            batch,
            batch_count_per_file,
            loss,
            need_update_grad,
            self.prompt_on_batch,
            self.train_args.llama_config.max_position_embeddings
        )

    def _on_epoch_end(self, epoch: int, loss, need_update_grad: bool):
        on_epoch_end(
            self.eval_model,
            epoch,
            loss,
            need_update_grad,
            self.prompt_on_epoch,
            self.train_args.llama_config.max_position_embeddings
        )

    def train(self):
        gradient_accumulation_steps = self.train_args.gradient_accumulation_steps
        global_steps = 0
        skipping_train = False

        for epoch in range(self.train_args.n_epochs):
            batch_count_per_epoch = 0
            loss_accumulation = torch.tensor(0.0, device=TrainerTools().parallel.device)
            self.train_model.train()

            for file_path in self.train_args.all_files:
                dataset = self._create_dataset(file_path)
                train_data_loader = TrainerTools().parallel.create_dataloader(
                    dataset=dataset,
                    data_loader_kwargs=self.data_loader_kwargs,
                    sampler_kwargs=self.sampler_kwargs
                )

                last_ckpt_batch = 0
                batch_count_per_file = len(train_data_loader)
                batch_count_per_epoch += batch_count_per_file

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
                        if TrainerTools().parallel.is_main_process and need_update_grad:
                            log(f"need_update_grad: {batch}")
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

                        self.scalar.scale(loss).backward()

                        if need_update_grad:
                            # clip grad
                            self.scalar.unscale_(self.optimizer)

                            if self.lr_scheduler.can_clip_grad():
                                torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)

                            self.lr_scheduler.step()

                            self.scalar.step(self.optimizer)
                            # optimizer.step()
                            self.scalar.update()
                            # flush the gradients as soon as we can, no need for this memory anymore
                            self.optimizer.zero_grad(set_to_none=True)

                            TrainerTools().parallel.synchronize()

                        loss_accumulation += loss.detach()

                    except Exception as e:
                        self._on_exception(e, epoch, batch)
                    finally:
                        if need_update_grad and (batch - last_ckpt_batch) >= self.train_args.eval_batch_interval:
                            last_ckpt_batch = batch
                            save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                            save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                            loss_item = loss.detach().item()
                            self._on_batch_end(
                                epoch,
                                batch,
                                batch_count_per_file,
                                loss_item * gradient_accumulation_steps if gradient_accumulation_steps > 1 else loss_item,
                                need_update_grad
                            )

                        try:
                            del loss
                        except UnboundLocalError:
                            pass

                        global_steps += 1

            if not skipping_train:
                if TrainerTools().parallel.parallel_train:
                    dist.all_reduce(loss_accumulation, dist.ReduceOp.AVG)

                TrainerTools().parallel.on_epoch_end(epoch)
                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                self._on_epoch_end(
                    epoch,
                    loss_accumulation.detach().item() / batch_count_per_epoch,
                    need_update_grad
                )

        TrainerTools().parallel.destroy()

"""
todo: 
0. 实现按照token数据确定batch大小，不固定batch的方案
1. 处理异常重启
2. 调研fsdp2 没有太多资料
3. Yarn和phi3的Phi3LongRoPEScaledRotaryEmbedding调研
4. MLA调研
5. DPO调研
6. inference使用缓存model，每个进程缓存一个
7. 多模态
"""
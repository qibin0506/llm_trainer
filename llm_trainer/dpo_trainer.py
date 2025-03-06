import time
import os
from typing import Tuple, List
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.distributed as dist

from llama import LlamaModel

from .parallel_ds import DsParallel
from .parallel_fsdp import FsdpParallel
from .trainer import Trainer
from .train_configs import TrainConfig
from .dataset import DPODataset
from .loss import DPOLoss
from .tools import TrainerTools
from .utils import dpo_collate_fn

from .checkpoint import (
    save_checkpoint,
    load_checkpoint_for_eval,
    save_steps,
)

from .trainer_log import (
    on_file_start,
)

class DPOTrainer(Trainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str]
    ):
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts
        )

        self.reference_model = self._init_reference_model()

    def _init_reference_model(self):
        parallel = TrainerTools().new_parallel()

        reference_model = LlamaModel(self.train_config.llama_config)
        load_checkpoint_for_eval(model=reference_model, device=parallel.device)

        reference_model, _ = parallel.process(
            model=reference_model,
            optimizer=None,
            kwargs=self._init_reference_args()
        )

        parallel.raw_model.eval()
        for param in parallel.raw_model.parameters():
            param.requires_grad = False

        return reference_model

    def _init_reference_args(self):
        if isinstance(TrainerTools().parallel, DsParallel) and self.train_config.ds_config:
            parallel_kwargs = {
                'gradient_accumulation_steps': 1,
                'train_micro_batch_size_per_gpu': 1
            }

            if self.train_config.ds_config.zero_config:
                zero_optimization = {'stage': 0}
                parallel_kwargs['zero_optimization'] = zero_optimization

            if self.train_config.ds_config.fp16_config:
                fb16_config = self.train_config.ds_config.fp16_config
                fp16 = { 'enabled': fb16_config.enabled }

                if fb16_config.fp16_opt_level is not None:
                    fp16['fp16_opt_level'] = fb16_config.fp16_opt_level

                parallel_kwargs['fp16'] = fp16

            if self.train_config.ds_config.bf16_config:
                bf16_config = self.train_config.ds_config.bf16_config
                bf16 = { 'enabled': bf16_config.enabled }
                parallel_kwargs['bf16'] = bf16
        elif isinstance(TrainerTools().parallel, FsdpParallel) and self.train_config.fsdp_config:
            parallel_kwargs = {
                'transformer_layer_cls': self.train_config.fsdp_config.transformer_layer_cls,
                'wrap_policy_num_params': self.train_config.fsdp_config.wrap_policy_num_params,
                'cpu_offload': self.train_config.fsdp_config.cpu_offload,
                'offload_params': self.train_config.fsdp_config.offload_params
            }
        else:
            parallel_kwargs = None

        return parallel_kwargs

    def _init_loss(self):
        criterion = DPOLoss(
            beta=self.train_config.dpo_loss_config.beta,
            label_smoothing=self.train_config.dpo_loss_config.label_smoothing,
            ipo=self.train_config.dpo_loss_config.ipo
        )

        return criterion, None

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": dpo_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_path) -> Dataset:
        max_position_embeddings = self.train_config.llama_config.max_position_embeddings
        return DPODataset(file_path, max_position_embeddings)

    def _calc_loss(self, inputs, attention_mask, logits, labels):
        pass

    def train(self):
        # 梯度累积步数
        gradient_accumulation_steps = self.train_config.gradient_accumulation_steps
        global_steps = 0
        loss_accumulation = 0.0
        skipping_train = False

        nll_loss_coef = self.train_config.dpo_loss_config.nll_loss_coef
        aux_loss_coef = self.train_config.loss_config.aux_loss_coef

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

                for batch, batch_data in enumerate(train_data_loader):
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
                        chosen_inputs = batch_data['chosen_inputs'].to(TrainerTools().parallel.device)
                        chosen_labels = batch_data['chosen_labels'].to(TrainerTools().parallel.device)
                        rejected_inputs = batch_data['rejected_inputs'].to(TrainerTools().parallel.device)
                        rejected_labels = batch_data['rejected_labels'].to(TrainerTools().parallel.device)

                        chosen_attention_mask = chosen_inputs != TrainerTools().tokenizer.pad
                        rejected_attention_mask = rejected_inputs != TrainerTools().tokenizer.pad

                        if TrainerTools().parallel.parallel_train:
                            # in DDP training we only need to sync gradients at the last micro step.
                            # the official way to do this is with model.no_sync() context manager, but
                            # I really dislike that this bloats the code and forces us to repeat code
                            # looking at the source of that context manager, it just toggles this variable
                            self.train_model.require_backward_grad_sync = need_update_grad

                        with self.ctx:
                            chosen_policy_result = self.train_model(chosen_inputs, attention_mask=chosen_attention_mask)
                            rejected_policy_result = self.train_model(rejected_inputs, attention_mask=rejected_attention_mask)

                            with torch.no_grad():
                                chosen_reference_result = self.reference_model(chosen_inputs, attention_mask=chosen_attention_mask)
                                rejected_reference_result = self.reference_model(rejected_inputs, attention_mask=rejected_attention_mask)

                            chosen_policy_logprobs, nll_loss = self.criterion.logprobs(chosen_policy_result['logits'], chosen_labels, chosen_attention_mask)
                            rejected_policy_logprobs, _ = self.criterion.logprobs(rejected_policy_result['logits'], rejected_labels, rejected_attention_mask)

                            chosen_reference_logprobs, _ = self.criterion.logprobs(chosen_reference_result['logits'], chosen_labels, chosen_attention_mask)
                            rejected_reference_logprobs, _ = self.criterion.logprobs(rejected_reference_result['logits'], rejected_labels, rejected_attention_mask)

                            # calc loss
                            loss, chosen_rewards, rejected_rewards = self.criterion(
                                chosen_policy_logprobs,
                                rejected_policy_logprobs,
                                chosen_reference_logprobs,
                                rejected_reference_logprobs
                            )

                            if nll_loss_coef:
                                loss += nll_loss_coef * nll_loss

                            if aux_loss_coef:
                                if chosen_policy_result['aux_loss']:
                                    loss += aux_loss_coef * chosen_policy_result['aux_loss']

                                if rejected_policy_result['aux_loss']:
                                    loss += aux_loss_coef * rejected_policy_result['aux_loss']

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


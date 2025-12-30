from typing import Optional, Tuple, List, Dict, Any
import copy
import gc
import importlib.metadata
from packaging import version

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from llm_model import LlmModel

from .parallel import DsParallel
from .tools import TrainerTools
from .loss import LMLoss, KDLoss
from .eval import submit_gen_task
from .partition_utils import unwrap_model_for_generation

from .train_configs import (
    TrainConfig,
    DsZero2Config,
    DsZero3Config,
    KDConfig
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
    set_seed,
    autocast,
)

from .log import Logger

class BaseTrainer:
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            kd_config: Optional[KDConfig] = None,
            gradient_accumulation_steps: int = 1
    ):
        set_seed()

        self.train_config: TrainConfig = train_config
        self.eval_prompts = eval_prompts
        self.eval_idx = -1
        self.last_global_steps = 0
        self.kd_config = kd_config
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.logger = Logger('log.txt')

        self.parallel_kwargs, self.data_loader_kwargs, self.sampler_kwargs = self._convert_train_args()
        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scaler = torch.GradScaler(enabled=TrainerTools().use_amp)

        # 注意：学习率要根据GPU的数量进行倍增：
        # 在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
        initial_lr = train_config.optim_config.initial_lr

        self.train_model, self.optimizer = self._init_train_model_and_optim(initial_lr)
        self.lr_scheduler = self._init_lr_scheduler(initial_lr, self.optimizer)

        self.criterion, self.kd_loss = self._init_loss()

        load_checkpoint(
            self.train_model,
            optimizer=self.optimizer,
            device=TrainerTools().parallel.device
        )

        steps_dict = load_steps()
        self._apply_restore_ckpt(steps_dict)

    def _new_model(self, train_config: TrainConfig):
        return LlmModel(train_config.model_config)

    def _init_train_model_and_optim(self, initial_lr: float):
        model = self._new_model(self.train_config)

        if self.train_config.init_state_dict:
            model.load_state_dict(self.train_config.init_state_dict, strict=False)
            self.train_config.init_state_dict = None

        self._check_freeze_llm_model(model)

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
        optimizer = None
        use_lion_optim = self.train_config.optim_config.optim_type == 'lion'

        if isinstance(TrainerTools().parallel, DsParallel) and self.parallel_kwargs:
            import deepspeed
            if ('zero_optimization' in self.parallel_kwargs
                    and 'offload_optimizer' in self.parallel_kwargs['zero_optimization']
                    and self.parallel_kwargs['zero_optimization']['offload_optimizer']['device'] == 'cpu'):
                if self.train_config.optim_config.optim_type == 'lion':
                    if version.parse(importlib.metadata.version("deepspeed")) >= version.parse('0.17.6'):
                        optimizer = deepspeed.ops.lion.DeepSpeedCPULion
                    else:
                        optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam
                        use_lion_optim = False
                        if TrainerTools().parallel.is_main_process:
                            Logger.std_log('When set offload_optimizer, lion optim is unsupported, so set optim to adam!!!!!')
                else:
                    optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam
            else:
                if self.train_config.optim_config.optim_type == 'lion':
                    optimizer = deepspeed.ops.lion.FusedLion
                else:
                    optimizer = deepspeed.ops.adam.FusedAdam

        if not optimizer:
            if self.train_config.optim_config.optim_type == 'lion':
                try:
                    import lion_pytorch
                except:
                    raise Exception('lion is not detected, please use `pip3 install lion_pytorch` to install or set optim_type to adam')

                optimizer = lion_pytorch.Lion
            else:
                optimizer = torch.optim.AdamW

        betas = self.train_config.optim_config.betas
        weight_decay = self.train_config.optim_config.weight_decay

        if betas is None:
            if use_lion_optim:
                betas = (0.95, 0.98)
            else:
                betas = (0.9, 0.999)

        if weight_decay is None:
            if use_lion_optim:
                weight_decay = 0.015
            else:
                weight_decay = 0.01

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

        return optimizer(
            optimizer_grouped_parameters,
            lr=initial_lr,
            betas=betas,
            weight_decay=weight_decay
        )

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

    def _apply_restore_ckpt(self, steps_dict):
        if steps_dict:
            self.last_global_steps = steps_dict['global_steps']
            if not self.last_global_steps:
                self.last_global_steps = 0

            self.lr_scheduler.restore_ckpt_dict(steps_dict)

            if TrainerTools().parallel.is_main_process:
                Logger.std_log(f'restore steps_dict={steps_dict}')

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs: Optional[Dict[str, Any]] = None
        if isinstance(TrainerTools().parallel, DsParallel) and self.train_config.ds_config:
            parallel_kwargs = {
                'gradient_accumulation_steps': 1,
                'gradient_clipping': self.train_config.ds_config.gradient_clipping,
                'train_micro_batch_size_per_gpu': self.train_config.batch_size
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

                if isinstance(zero_config, DsZero2Config) or isinstance(zero_config, DsZero3Config):
                    if zero_config.offload_optimizer is not None:
                        zero_optimization['offload_optimizer'] = {
                            "device": zero_config.offload_optimizer.device,
                            "pin_memory": zero_config.offload_optimizer.pin_memory
                        }
                    if zero_config.offload_param is not None:
                        zero_optimization['offload_param'] = {
                            "device": zero_config.offload_param.device,
                            "pin_memory": zero_config.offload_param.pin_memory
                        }

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

            if (self.train_config.ds_config.bf16_config is not None
                    and self.train_config.ds_config.bf16_config.enabled):
                bf16_config = self.train_config.ds_config.bf16_config
                bf16 = {
                    'enabled': bf16_config.enabled
                }
                parallel_kwargs['bf16'] = bf16
            elif self.train_config.ds_config.fp16_config:
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
            "pin_memory": dataloader_args.data_loader_pin_memory,
            "num_workers": dataloader_args.data_loader_num_workers,
            "shuffle": dataloader_args.data_loader_shuffle,
            "drop_last": dataloader_args.data_loader_drop_last,
        }
        sampler_kwargs = {
            "shuffle": dataloader_args.data_loader_shuffle,
            "drop_last": dataloader_args.data_loader_drop_last,
        }

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _init_ref_model_args(self) -> dict:
        parallel_kwargs = copy.deepcopy(self.parallel_kwargs) if self.parallel_kwargs else None

        if parallel_kwargs and isinstance(TrainerTools().parallel, DsParallel):
            # reference to https://github.com/huggingface/trl/blob/main/trl/models/utils.py:prepare_deepspeed
            # if model is not None:
            #     hidden_size = (
            #         max(model.config.hidden_sizes)
            #         if getattr(model.config, "hidden_sizes", None)
            #         else getattr(model.config, "hidden_size", None)
            #     )
            #     if hidden_size is not None and stage == 3:
            #         # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            #         # @ step 0: expected module 1, but got module 0`
            #         # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            #         config_kwargs.update(
            #             {
            #                 "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
            #                 "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
            #                 "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
            #             }
            #         )

            parallel_kwargs.pop('activation_checkpointing', None)
            parallel_kwargs.pop('gradient_clipping', None)

            # ref_model暂时先使用stage 0, 解决训练卡住问题
            parallel_kwargs["zero_optimization"] = {"stage": 0}
            # if parallel_kwargs.get("zero_optimization", {}).get("stage", 0) != 3:
            #     parallel_kwargs["zero_optimization"] = {"stage": 0}

        return parallel_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]: ...

    def _calc_loss(self, inputs, attention_mask, logits, labels):
        # calc loss
        if not self.kd_loss or self.kd_config.kd_coef == 0.0:
            # 不用计算kd_loss
            return self.criterion(logits, labels)

        teacher_logits = self.kd_config.teacher_logits_provider(inputs, attention_mask)
        loss = self.kd_loss(logits, teacher_logits, labels)

        if self.kd_config.kd_coef == 1.0:
            # 不用计算ce loss
            return loss

        ce_loss = self.criterion(logits, labels)
        return (1 - self.kd_config.kd_coef) * ce_loss + self.kd_config.kd_coef * loss

    def _backward_loss(self, loss):
        if isinstance(TrainerTools().parallel, DsParallel):
            self.train_model.backward(loss)
        else:
            self.scaler.scale(loss).backward()

    def _apply_grad_clipping(self):
        # ds模式已经集成gradient_clipping
        if not isinstance(TrainerTools().parallel, DsParallel) and self.lr_scheduler.can_clip_grad():
            # clip grad
            self.scaler.unscale_(self.optimizer)

            trainable_params = filter(lambda p: p.requires_grad, self.train_model.parameters())
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

    def _apply_step(self):
        self.lr_scheduler.step()
        if isinstance(TrainerTools().parallel, DsParallel):
            self.train_model.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

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

    def _log(self, keys: Dict[str, any], values: Dict[str, any]):
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

    def _get_model_dtype(self):
        if isinstance(TrainerTools().parallel, DsParallel):
            import deepspeed
            assert isinstance(self.train_model, deepspeed.DeepSpeedEngine)
            return self.train_model.get_data_types()[0]
        else:
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    def _eval(self, tag: str):
        with unwrap_model_for_generation(self.train_model) as eval_model:
            if TrainerTools().parallel.is_main_process:
                eval_prompt = self._get_eval_data()

                if eval_prompt:
                    eval_model = self._check_eval_model(eval_model)
                    eval_model.eval()

                    eval_pixel_values, tokens_per_image = self._get_eval_pixel_values_and_tokens_count(self.eval_idx)
                    submit_gen_task(
                        eval_model,
                        self.train_config,
                        tag=tag,
                        prompt=eval_prompt,
                        pixel_values=eval_pixel_values,
                        tokens_per_image=tokens_per_image
                    )

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
            gradient_accumulation_steps,
            batches_accumulated
    ) -> List[float]:
        avg_losses = []
        for loss in losses:
            avg_loss = torch.tensor(
                loss * gradient_accumulation_steps / batches_accumulated,
                device=TrainerTools().parallel.device)

            if TrainerTools().parallel.parallel_train:
                dist.all_reduce(avg_loss, dist.ReduceOp.AVG)

            avg_losses.append(avg_loss.detach().item())

        return avg_losses

    def _get_pixel_values(self, batch_data):
        return None

    def train(self):
        # 梯度累积步数
        gradient_accumulation_steps = max(1, self.gradient_accumulation_steps)
        global_steps = 0
        skipping_train = False

        loss_accumulation = 0.0
        aux_loss_accumulation = 0.0
        batches_accumulated = 0

        for epoch in range(self.train_config.n_epochs):
            self.train_model.train()
            file_count = len(self.train_config.file_dataset)

            for file_idx in range(file_count):
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

                for batch, batch_data in enumerate(train_data_loader):
                    global_steps += 1
                    if global_steps < self.last_global_steps:
                        skipping_train = True
                        continue

                    # 是否需要更新梯度
                    if skipping_train:
                        need_update_grad = False
                    elif gradient_accumulation_steps > 1:
                        need_update_grad = (batch + 1) % gradient_accumulation_steps == 0 or batch == batch_count_per_file - 1
                    else:
                        need_update_grad = True

                    # 要放在need_update_grad赋值下面，解决在继续训练时未知原因的卡死现象
                    if skipping_train:
                        TrainerTools().parallel.wait('skip train')
                        skipping_train = False

                    inputs = batch_data['inputs']
                    labels = batch_data['labels']

                    try:
                        inputs, labels = inputs.to(TrainerTools().parallel.device), labels.to(TrainerTools().parallel.device)
                        attention_mask = inputs != TrainerTools().tokenizer.pad
                        pixel_values = self._get_pixel_values(batch_data)

                        if TrainerTools().parallel.parallel_train:
                            self.train_model.require_backward_grad_sync = need_update_grad

                        with autocast(TrainerTools().parallel.device_type):
                            result = self.train_model(
                                inputs,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values
                            )

                            # calc loss
                            loss = self._calc_loss(inputs, attention_mask, result['logits'], labels)
                            if result['aux_loss'] and self.train_config.loss_config.aux_loss_coef:
                                aux_loss = self.train_config.loss_config.aux_loss_coef * result['aux_loss']
                            else:
                                aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps
                            aux_loss = aux_loss / gradient_accumulation_steps

                        total_loss = loss + aux_loss
                        self._backward_loss(total_loss)

                        loss_accumulation += total_loss.detach().item()
                        aux_loss_accumulation += aux_loss.detach().item()

                        batches_accumulated += 1

                        if need_update_grad:
                            self._apply_grad_clipping()
                            self._apply_step()

                            avg_loss, avg_aux_loss = self._avg_loss(
                                losses=[
                                    loss_accumulation,
                                    aux_loss_accumulation
                                ],
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                batches_accumulated=batches_accumulated
                            )

                            self._log(
                                keys={
                                    'epoch': epoch,
                                    'file': f'{file_idx + 1}/{file_count}',
                                    'batch': f'{batch}/{batch_count_per_file}'
                                },
                                values={
                                    'loss': avg_loss,
                                    'moe_aux_loss': avg_aux_loss
                                }
                            )

                            # reset to default
                            loss_accumulation = 0.0
                            aux_loss_accumulation = 0.0
                            batches_accumulated = 0
                    except Exception as e:
                        self._on_exception(e, epoch, batch)
                    finally:
                        if need_update_grad:
                            save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                            if (batch - last_ckpt_batch) >= self.train_config.eval_config.eval_batch_interval:
                                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                                last_ckpt_batch = batch
                                self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')

                # 一个文件训练结束后，清理内存
                del train_data_loader
                del dataset
                if hasattr(TrainerTools().parallel, '_sampler'):
                    TrainerTools().parallel._sampler = None

                gc.collect()
                torch.cuda.empty_cache()

            # end epoch
            if not skipping_train:
                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                save_checkpoint(model=self.train_model, optimizer=self.optimizer)

                TrainerTools().parallel.on_epoch_end(epoch)
                self._on_epoch_end(tag=f'epoch:{epoch}')

        TrainerTools().parallel.destroy()

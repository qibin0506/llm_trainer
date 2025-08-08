from typing import Optional, Tuple, List, Dict, Any
import copy

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from llm_model import LlmModel, VlmModel

from .parallel_ds import DsParallel
from .tools import TrainerTools
from .loss import LMLoss, KDLoss
from .dataset import TextDataset
from .eval import submit_gen_task
from .partition_utils import unwrap_model_for_generation

from .train_configs import (
    TrainConfig,
    VLMConfig,
    DsZero2Config,
    DsZero3Config
)

from .scheduler import (
    LRScheduler,
    WarmupCosineAnnealingLRScheduler,
    NoneLRScheduler
)

from .checkpoint import (
    load_checkpoint,
    save_checkpoint,
    save_best_checkpoint,
    load_steps,
    save_steps,
)

from .utils import (
    set_seed,
    autocast,
    create_doc_boundary_mask,
    generate_position_ids,
    pretrain_collate_fn,
)

from .log import(
    log,
    get_log_dir
)

class Trainer:
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            eval_image_tags: Optional[List[str]] = None
    ):
        set_seed()

        # 是否打包序列，仅pretrain阶段需要打包序列，
        # [[1, 1, eos, 2, 2, eos]]
        #   doc_boundary_mask=[[[[0., 0., 0., 0., 0., 0.],
        #           [0., 0., 0., 0., 0., 0.],
        #           [0., 0., 0., 0., 0., 0.],
        #           [-inf, -inf, -inf, 0., 0., 0.],
        #           [-inf, -inf, -inf, 0., 0., 0.],
        #           [-inf, -inf, -inf, 0., 0., 0.]]]]
        #   position_ids=[[0, 1, 2, 0, 1, 2]]
        self.packed_sequences = True

        self.train_config: TrainConfig = train_config
        self.eval_prompts = eval_prompts
        self.eval_image_tags = eval_image_tags
        self.eval_idx = -1
        self.last_global_steps = 0

        if self.eval_image_tags:
            assert len(self.eval_prompts) == len(self.eval_image_tags)

        parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim = self._convert_train_args()
        self.parallel_kwargs = parallel_kwargs
        self.data_loader_kwargs: dict[str, Any] = data_loader_kwargs
        self.sampler_kwargs: dict[str, Any] = sampler_kwargs

        # initialize a GradScaler. If enabled=False scaler is a no-op
        self.scalar = torch.GradScaler(enabled=TrainerTools().use_amp)

        # 注意：学习率要根据GPU的数量进行倍增：
        # 在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
        initial_lr = train_config.lr_config.initial_lr

        self.train_model, self.optimizer = self._init_train_model_and_optim(initial_lr, parallel_kwargs, use_ds_optim)
        self.lr_scheduler = self._init_lr_scheduler(initial_lr)

        self.criterion, self.kd_loss = self._init_loss()

        load_checkpoint(
            self.train_model,
            optimizer=self.optimizer,
            device=TrainerTools().parallel.device
        )

        steps_dict = load_steps()
        if steps_dict:
            self.last_global_steps = steps_dict['global_steps']
            if not self.last_global_steps:
                self.last_global_steps = 0

            self.lr_scheduler.restore_ckpt_dict(steps_dict)

            log(f'restore steps_dict = {steps_dict}')

        if isinstance(train_config.model_config, VLMConfig):
            self.pixel_values_provider = train_config.pixel_values_provider
            self.tokens_per_image = train_config.model_config.tokens_per_image
        else:
            self.pixel_values_provider = None
            self.tokens_per_image = -1

    def _new_model(self, train_config: TrainConfig):
        if isinstance(train_config.model_config, VLMConfig):
            return VlmModel(train_config.model_config)
        else:
            return LlmModel(train_config.model_config)

    def _get_trainable_params(self, model):
        freeze_llm_model = self.train_config.freeze_llm_model
        return model.parameters() if not freeze_llm_model else filter(lambda p: p.requires_grad, model.parameters())

    def _init_train_model_and_optim(
            self,
            initial_lr: float,
            parallel_kwargs: dict,
            use_ds_optim: bool
    ):
        model = self._new_model(self.train_config)

        if self.train_config.init_state_dict:
            model.load_state_dict(self.train_config.init_state_dict, strict=False)
            self.train_config.init_state_dict = None

        # freeze llm model for vlm training
        if self.train_config.freeze_llm_model:
            for name, param in model.named_parameters():
                if not any(sub_module in name for sub_module in ['multi_modal_projector']):
                    param.requires_grad = False

            # model.embed_tokens.eval()
            # model.layers.eval()
            # model.head_norm.eval()
            # model.lm_head.eval()

        if TrainerTools().parallel.is_main_process:
            total_params = sum(p.numel() for p in model.parameters())
            log(f"Total number of parameters: {total_params:,}")

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log(f"Trainable number of parameters: {trainable_params:,}")

            total_size_bytes = total_params * 4
            total_size_mb = total_size_bytes / (1024 * 1024)
            log(f"Total size of the model: {total_size_mb:.2f} MB")

        if use_ds_optim:
            import deepspeed
            origin_optim = deepspeed.ops.adam.DeepSpeedCPUAdam(
                self._get_trainable_params(model),
                lr=initial_lr,
                weight_decay=self.train_config.lr_config.weight_decay
            )
        else:
            origin_optim = torch.optim.AdamW(
                self._get_trainable_params(model),
                lr=initial_lr,
                weight_decay=self.train_config.lr_config.weight_decay
            )
        model, optim = TrainerTools().parallel.process(
            model=model,
            optimizer=origin_optim,
            kwargs=parallel_kwargs
        )

        return model, optim

    def _init_lr_scheduler(self, initial_lr: float) -> LRScheduler:
        if self.train_config.lr_config.enable_lr_scheduler:
            warmup_iters = self.train_config.lr_config.warmup_iters
            min_lr = self.train_config.lr_config.min_lr
            max_lr = self.train_config.lr_config.max_lr
            cosine_annealing_period = self.train_config.lr_config.cosine_annealing_period
            cosine_annealing_period_mul = self.train_config.lr_config.cosine_annealing_period_mul

            return WarmupCosineAnnealingLRScheduler(
                optimizer=self.optimizer,
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

        kd_loss = KDLoss() if self.train_config.kd_config else None

        return criterion, kd_loss

    def _convert_train_args(self) -> Tuple[dict, dict, dict, bool]:
        parallel_kwargs: Optional[Dict[str, Any]] = None
        use_ds_optim: bool = False
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
                        use_ds_optim = True
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
            "collate_fn": pretrain_collate_fn,
            "num_workers": dataloader_args.data_loader_num_workers,
            "shuffle": dataloader_args.data_loader_shuffle,
            "drop_last": dataloader_args.data_loader_drop_last,
        }
        sampler_kwargs = {
            "shuffle": dataloader_args.data_loader_shuffle,
            "drop_last": dataloader_args.data_loader_drop_last,
        }

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim

    def _init_ref_model_args(self) -> dict:
        parallel_kwargs = copy.deepcopy(self.parallel_kwargs)

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

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        max_position_embeddings = self.train_config.model_config.max_position_embeddings
        return TextDataset(file_path, max_position_embeddings, max_position_embeddings), file_path

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

    def _get_eval_data(self) -> Tuple[str, Optional[str]]:
        if len(self.eval_prompts) == 0:
            return '', None

        self.eval_idx += 1
        if self.eval_idx == len(self.eval_prompts):
            self.eval_idx = 0

        if not self.eval_image_tags:
            return self.eval_prompts[self.eval_idx], None

        return self.eval_prompts[self.eval_idx], self.eval_image_tags[self.eval_idx]

    def _log_loss(
            self,
            epoch_tag: str,
            file_tag: str,
            batch_tag: str,
            loss
    ):
        if TrainerTools().parallel.is_main_process:
            log_dir = get_log_dir()
            log_msg = f"{epoch_tag}, {file_tag}, {batch_tag}, loss: {loss}"
            log(log_msg)
            log(f"{log_msg}\n", f'{log_dir}log.txt')

    def _on_exception(
            self,
            e: Exception,
            epoch: int,
            batch: int
    ):
        log_dir = get_log_dir()
        exception_file = e.__traceback__.tb_frame.f_globals["__file__"]
        exception_line = e.__traceback__.tb_lineno
        log_msg = f"epoch: {epoch}, batch: {batch}, {e} at {exception_file} line {exception_line}\n"
        log(log_msg, f'{log_dir}log.txt')

        raise e

    def _get_model_dtype(self):
        if isinstance(TrainerTools().parallel, DsParallel):
            import deepspeed
            assert isinstance(self.train_model, deepspeed.DeepSpeedEngine)
            return self.train_model.get_data_types()[0]
        else:
            return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    def _eval(self, tag: str):
        with unwrap_model_for_generation(self.train_model) as generate_model:
            if TrainerTools().parallel.is_main_process:
                generate_model.eval()
                eval_prompt, eval_image_tag = self._get_eval_data()

                if isinstance(self.train_config, VLMConfig) and self.pixel_values_provider and eval_image_tag:
                    eval_pixel_values = self.pixel_values_provider([eval_image_tag])
                else:
                    eval_pixel_values = None

                submit_gen_task(
                    generate_model,
                    self.train_config.eval_config,
                    tag=tag,
                    prompt=eval_prompt,
                    pixel_values=eval_pixel_values,
                    max_position_embeddings=self.train_config.model_config.max_position_embeddings,
                    tokens_per_image=self.tokens_per_image
                )
                generate_model.train()

        TrainerTools().parallel.wait('eval')

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
            log(f"epoch: {epoch}, start train {file_name}\n", f'{get_log_dir()}log.txt')

    def train(self):
        # 梯度累积步数
        gradient_accumulation_steps = self.train_config.gradient_accumulation_steps
        global_steps = 0
        skipping_train = False

        loss_accumulation = 0.0
        batches_accumulated = 0
        current_loss: float = 0.0
        last_best_checkpoint_loss: Optional[float] = None

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

                        if self.packed_sequences:
                            doc_boundary_mask = create_doc_boundary_mask(inputs, self._get_model_dtype())
                            position_ids = generate_position_ids(inputs)
                        else:
                            doc_boundary_mask = None
                            position_ids = None

                        if self.pixel_values_provider and 'image_tags' in batch_data:
                            image_tags = batch_data['image_tags']
                            pixel_values = self.pixel_values_provider(image_tags).to(TrainerTools().parallel.device)
                        else:
                            pixel_values = None

                        if TrainerTools().parallel.parallel_train:
                            self.train_model.require_backward_grad_sync = need_update_grad

                        with autocast(TrainerTools().parallel.device_type):
                            result = self.train_model(
                                inputs,
                                attention_mask=attention_mask,
                                doc_boundary_mask=doc_boundary_mask,
                                position_ids=position_ids,
                                pixel_values=pixel_values
                            )

                            # calc loss
                            loss = self._calc_loss(inputs, attention_mask, result['logits'], labels)
                            if result['aux_loss'] and self.train_config.loss_config.aux_loss_coef:
                                loss += self.train_config.loss_config.aux_loss_coef * result['aux_loss']

                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                        loss_accumulation += loss.detach().item()
                        self._backward_loss(loss)
                        batches_accumulated += 1

                        if need_update_grad:
                            loss_tensor = torch.tensor(loss_accumulation * gradient_accumulation_steps / batches_accumulated, device=TrainerTools().parallel.device)

                            if TrainerTools().parallel.parallel_train:
                                dist.all_reduce(loss_tensor, dist.ReduceOp.AVG)

                            current_loss = loss_tensor.item()

                            # ds模式已经集成gradient_clipping
                            if not isinstance(TrainerTools().parallel, DsParallel) and self.lr_scheduler.can_clip_grad():
                                # clip grad
                                self.scalar.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self._get_trainable_params(self.train_model), 1.0)

                            self._step()

                            self._log_loss(
                                epoch_tag=f'epoch: {epoch}',
                                file_tag=f'file: {file_idx + 1}/{file_count}',
                                batch_tag=f'batch: {batch}/{batch_count_per_file}',
                                loss=current_loss
                            )
                            # reset to default
                            loss_accumulation = 0.0
                            batches_accumulated = 0
                    except Exception as e:
                        self._on_exception(e, epoch, batch)
                    finally:
                        if need_update_grad:
                            save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                            if (batch - last_ckpt_batch) >= self.train_config.eval_batch_interval:
                                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                                if save_best_checkpoint(current_loss, last_best_checkpoint_loss):
                                    last_best_checkpoint_loss = current_loss

                                last_ckpt_batch = batch
                                self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')

                        try:
                            del loss
                        except UnboundLocalError: ...

            # end epoch
            if not skipping_train:
                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                if save_best_checkpoint(current_loss, last_best_checkpoint_loss):
                    last_best_checkpoint_loss = current_loss

                TrainerTools().parallel.on_epoch_end(epoch)
                self._on_epoch_end(tag=f'epoch:{epoch}')

        TrainerTools().parallel.destroy()

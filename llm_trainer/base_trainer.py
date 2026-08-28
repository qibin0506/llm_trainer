from typing import Optional, Tuple, List, Dict, Any
import os
import copy
import gc
import math
import importlib.metadata
from packaging import version
from itertools import islice
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from llm_model import LlmModel, ModelConfig

from .parallel import DsParallel
from .tools import TrainerTools
from .partition_utils import unwrap_model_for_generation
from .generate_utils import generate
from .log import Logger, _get_log_dir
from .train_configs import TrainConfig, GenerationService
from .scheduler import LRScheduler, WarmupCosineAnnealingLRScheduler, NoneLRScheduler
from .checkpoint import load_checkpoint, save_checkpoint, load_steps, save_steps
from .utils import (
    default_seed,
    set_seed,
    autocast,
    is_bf16_supported,
    empty_cache,
    _build_deepspeed_kwargs,
    _build_data_loader_config
)
from .muon import MuonWithAdamW, _get_ds_muon


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

            gradient_accumulation_steps:
                - 梯度累积步数，用于通过累积多批数据的梯度来模拟更大的 Global Batch Size。

            return_logits:
                - 模型forward的时候是否返回logits
        """
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            generation_service: Optional[GenerationService] = None,
            gradient_accumulation_steps: int = 1,
            return_logits: bool = True
    ):
        set_seed(default_seed)

        self.is_ds = isinstance(TrainerTools().parallel, DsParallel)
        self.train_config: TrainConfig = train_config
        self.eval_prompts = eval_prompts
        self.generation_service = generation_service
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.return_logits = return_logits

        self.eval_idx = -1
        self.resume_epoch = 0
        self.resume_file_idx = 0
        self.resume_batch_idx = 0


        self.logger = Logger('log.txt')

        self.parallel_kwargs, self.data_loader_kwargs, self.sampler_kwargs = self._convert_train_args()
        # initialize a GradScaler. If enabled=False scaler is a no-op
        self._init_scaler()

        # 注意：学习率要根据GPU的数量进行倍增：
        # 在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
        initial_lr = train_config.optim_config.initial_lr

        self.train_model, self.optimizer = self._init_train_model_and_optim(initial_lr)
        self.lr_scheduler = self._init_lr_scheduler(initial_lr, self.optimizer)

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

    def _new_model_context(self, parallel_kwargs):
        if self.is_ds and parallel_kwargs and 'zero_optimization' in parallel_kwargs:
            stage = parallel_kwargs["zero_optimization"].get("stage", 0)
            if stage == 3:
                import deepspeed
                return deepspeed.zero.Init(config_dict_or_path=parallel_kwargs)

        return nullcontext()

    def _new_model(self, train_config: TrainConfig):
        return LlmModel(train_config.model_config)

    def _load_external_weights(
            self,
            model: torch.nn.Module,
            weights_path: str,
            prefixes: Optional[List[str]] = None
    ):
        max_elements_per_chunk = int(os.environ.get('LOAD_WEIGHTS_MAX_ELEMENTS_PER_CHUNK', 100_000_000))

        if TrainerTools().parallel.is_main_process:
            Logger.std_log(f"Loading external weights from {weights_path} ...")

        if os.path.isfile(weights_path):
            files = [weights_path]
        elif os.path.isdir(weights_path):
            st_files = [f for f in os.listdir(weights_path) if f.endswith('.safetensors')]
            if st_files:
                files = [os.path.join(weights_path, f) for f in st_files]
            else:
                pt_files = [f for f in os.listdir(weights_path) if f.endswith(('.bin', '.pt', '.pth'))]
                files = [os.path.join(weights_path, f) for f in pt_files]
        else:
            raise ValueError(f"Invalid weight_path: {weights_path}")

        if not files:
            raise FileNotFoundError(f"No valid weight files (.safetensors, .bin, .pt, .pth) found in {weights_path}")

        files.sort()

        is_zero3 = False
        try:
            import deepspeed
            if isinstance(model, deepspeed.DeepSpeedEngine):
                if hasattr(model, "zero_optimization_stage") and model.zero_optimization_stage() == 3:
                    is_zero3 = True
            elif any(hasattr(p, 'ds_id') for p in model.parameters()):
                is_zero3 = True
        except ImportError: ...

        target_model = model.module if hasattr(model, 'module') else model

        for f in files:
            state_dict = {}
            if (not is_zero3) or TrainerTools().parallel.is_main_process:
                if f.endswith('.safetensors'):
                    try:
                        from safetensors.torch import load_file
                    except ImportError:
                        raise ImportError("Please install safetensors: pip install safetensors")
                    state_dict = load_file(f, device="cpu")
                else:
                    state_dict = torch.load(f, map_location="cpu", weights_only=True)
                    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']

            if prefixes:
                mapped_state_dict = {}
                for k, v in state_dict.items():
                    for prefix in prefixes:
                        mapped_state_dict[prefix + k] = v
                state_dict = mapped_state_dict

            valid_keys = []
            if TrainerTools().parallel.is_main_process:
                valid_keys = list(state_dict.keys())

            if TrainerTools().parallel.world_size > 1:
                object_list = [valid_keys]
                torch.distributed.broadcast_object_list(object_list, src=0)
                valid_keys = object_list[0]

            if is_zero3:
                import deepspeed
                chunk_params = []
                chunk_names = []
                current_elements = 0

                def _flush_chunk():
                    nonlocal chunk_params, chunk_names, current_elements
                    if not chunk_params:
                        return

                    with deepspeed.zero.GatheredParameters(chunk_params, modifier_rank=0):
                        if TrainerTools().parallel.is_main_process:
                            for n, p in zip(chunk_names, chunk_params):
                                if n in state_dict:
                                    p.data.copy_(state_dict[n].to(p.device, dtype=p.dtype))

                    chunk_params.clear()
                    chunk_names.clear()
                    current_elements = 0

                with torch.no_grad():
                    for name, param in target_model.named_parameters():
                        if name in valid_keys:
                            chunk_params.append(param)
                            chunk_names.append(name)
                            current_elements += getattr(param, 'ds_numel', param.numel())

                            if current_elements >= max_elements_per_chunk:
                                _flush_chunk()

                    _flush_chunk()

                    for name, buf in target_model.named_buffers():
                        if name in valid_keys:
                            if TrainerTools().parallel.is_main_process:
                                buf.data.copy_(state_dict[name].to(buf.device, dtype=buf.dtype))
                            if TrainerTools().parallel.world_size > 1:
                                tmp_tensor = buf.data.to(TrainerTools().parallel.device).contiguous()
                                torch.distributed.broadcast(tmp_tensor, src=0)
                                buf.data.copy_(tmp_tensor.to(buf.device, non_blocking=True))
            else:
                target_model.load_state_dict(state_dict, strict=False)

            del state_dict
            import gc
            gc.collect()
            empty_cache()

        if TrainerTools().parallel.is_main_process:
            Logger.std_log("Successfully loaded weights.")

    def _init_train_model_and_optim(self, initial_lr: float):
        with self._new_model_context(self.parallel_kwargs):
            model = self._new_model(self.train_config)

        self._check_freeze_llm_model(model)

        if self.train_config.gradient_checkpointing:
            if self.is_ds:
                import deepspeed
                model.gradient_checkpointing_enable(checkpoint_func=deepspeed.checkpointing.checkpoint)
            else:
                model.gradient_checkpointing_enable()

        if TrainerTools().parallel.is_main_process:
            total_params = sum(getattr(p, 'ds_numel', p.numel()) for p in model.parameters())
            Logger.std_log(f"Total number of parameters: {total_params:,}")

            trainable_params = sum(getattr(p, 'ds_numel', p.numel()) for p in model.parameters() if p.requires_grad)
            Logger.std_log(f"Trainable number of parameters: {trainable_params:,}")

            total_size_bytes = total_params * 4
            total_size_mb = total_size_bytes / (1024 * 1024)
            Logger.std_log(f"Total size of the model: {total_size_mb:.2f} MB")

        if self.train_config.init_weights_path is not None:
            self._load_external_weights(model, self.train_config.init_weights_path)

        optimizer = self._config_optim(model, initial_lr)

        model, optim = TrainerTools().parallel.process(
            model=model,
            optimizer=optimizer,
            kwargs=self.parallel_kwargs
        )

        return model, optim

    def _check_freeze_llm_model(self, model): ...

    def _config_optim(self, model, initial_lr):
        optim_config = self.train_config.optim_config
        optim_cls = self._get_optim_cls()

        group_data = self._build_optimizer_param_groups(model, optim_config, name_prefix="base")
        if group_data["type"] == "muon":
            muon_wd = optim_config.muon_weight_decay
            if muon_wd is None:
                muon_wd = optim_config.weight_decay if optim_config.weight_decay is not None else 0.01

            if isinstance(TrainerTools().parallel, DsParallel):
                all_groups = group_data["muon_groups"] + group_data["adamw_groups"]
                ds_muon = _get_ds_muon(all_groups)

                if ds_muon is None:
                    raise RuntimeError('Current deepspeed is not support muon')
                return ds_muon

            return MuonWithAdamW(
                muon_params=group_data["muon_groups"],
                adamw_params=group_data["adamw_groups"],
                muon_lr=optim_config.muon_lr if optim_config.muon_lr is not None else 0.02,
                muon_momentum=optim_config.muon_momentum if optim_config.muon_momentum is not None else 0.95,
                muon_weight_decay=muon_wd,
                muon_ns_steps=optim_config.muon_ns_steps if optim_config.muon_ns_steps is not None else 5,
                muon_adjust_lr_fn=optim_config.muon_adjust_lr_fn,
                adamw_lr=initial_lr,
                adamw_betas=optim_config.betas if optim_config.betas is not None else (0.9, 0.95),
                adamw_weight_decay=optim_config.weight_decay if optim_config.weight_decay is not None else 0.01,
                adamw_cls=optim_cls
            )
        else:
            return optim_cls(
                group_data["groups"],
                lr=initial_lr,
                betas=optim_config.betas if optim_config.betas is not None else ((0.95, 0.98) if optim_config.optim_type == 'lion' else (0.9, 0.999)),
                weight_decay=optim_config.weight_decay if optim_config.weight_decay is not None else (0.015 if optim_config.optim_type == 'lion' else 0.01)
            )

    def _get_optim_cls(self):
        optimizer = None

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

        return optimizer

    def _build_optimizer_param_groups(self, module: torch.nn.Module, optim_config, name_prefix: str = ""):
        optim_type = optim_config.optim_type
        weight_decay = optim_config.weight_decay if optim_config.weight_decay is not None else (
            0.015 if optim_type == 'lion' else 0.01)
        lr = optim_config.initial_lr
        max_lr = optim_config.max_lr

        def _make_group(params, wd, group_name, custom_lr=lr, custom_max_lr=max_lr, custom_wd=None, use_muon=False, momentum=0.95):
            g = {
                "params": params,
                "weight_decay": wd if custom_wd is None else custom_wd,
                "lr": custom_lr,
                "max_lr": custom_max_lr,
                "use_muon": use_muon,
                "name": f"{name_prefix}_{group_name}"
            }
            if use_muon:
                g["momentum"] = momentum
            if optim_config.betas is not None:
                g["betas"] = optim_config.betas
            else:
                g["betas"] = (0.95, 0.98) if optim_type == 'lion' else (0.9, 0.999)
            return g

        if optim_type != 'muon':
            no_decay_name_list = ["bias", "norm.weight"]
            decay_params, no_decay_params = [], []
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                if any(nd in name for nd in no_decay_name_list):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            raw_groups = [
                _make_group(decay_params, weight_decay, "decay"),
                _make_group(no_decay_params, 0.0, "no_decay")
            ]
            return {
                "type": optim_type,
                "groups": [g for g in raw_groups if len(g["params"]) > 0]
            }

        muon_lr = optim_config.muon_lr if optim_config.muon_lr is not None else 0.02
        muon_momentum = optim_config.muon_momentum if optim_config.muon_momentum is not None else 0.95
        muon_wd = optim_config.muon_weight_decay if optim_config.muon_weight_decay is not None else weight_decay
        adam_keywords = ["embed", "wte", "wpe", "lm_head", "head", "output", "bias", "norm"]
        muon_params, adam_decay_params, adam_no_decay_params = [], [], []

        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            is_2d = (param.ndim == 2)
            is_special = any(kw in name.lower() for kw in adam_keywords)
            if is_2d and not is_special:
                muon_params.append(param)
            else:
                if "bias" in name.lower() or "norm" in name.lower():
                    adam_no_decay_params.append(param)
                else:
                    adam_decay_params.append(param)

        muon_groups = [
            _make_group(muon_params, muon_wd, "muon", custom_lr=muon_lr, custom_max_lr=muon_lr, use_muon=True,
                        momentum=muon_momentum)
        ]
        adamw_groups = [
            _make_group(adam_decay_params, weight_decay, "adam_decay", use_muon=False),
            _make_group(adam_no_decay_params, 0.0, "adam_no_decay", use_muon=False)
        ]

        return {
            "type": "muon",
            "muon_groups": [g for g in muon_groups if len(g["params"]) > 0],
            "adamw_groups": [g for g in adamw_groups if len(g["params"]) > 0]
        }

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
        ds_config = self.train_config.ds_config

        if isinstance(TrainerTools().parallel, DsParallel) and ds_config:
            parallel_kwargs = _build_deepspeed_kwargs(
                ds_config,
                self.gradient_accumulation_steps,
                self.train_config.batch_size,
                self.train_config.optim_config.optim_type
            )

        data_loader_kwargs, sampler_kwargs = _build_data_loader_config(
            self.train_config.data_loader_config,
            self.train_config.batch_size
        )

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

    def _calc_loss(self, inputs, attention_mask, result, labels) -> Tuple[torch.Tensor, Optional[torch.Tensor]]: ...

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

            if hasattr(self.train_model, 'policy_model') and hasattr(self.train_model, 'value_model'):
                policy_params = [p for p in self.train_model.policy_model.parameters() if p.requires_grad]
                value_params = [p for p in self.train_model.value_model.parameters() if p.requires_grad]

                if policy_params:
                    torch.nn.utils.clip_grad_norm_(policy_params, 1.0)
                if value_params:
                    torch.nn.utils.clip_grad_norm_(value_params, 1.0)
            else:
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
            tokens = torch.tensor(TrainerTools().tokenizer.encode(eval_prompt), dtype=torch.long).unsqueeze(0)
            service_output = self.generation_service(
                self.train_model, tokens, self.train_config.eval_config,
                'eval', eval_pixel_values, tokens_per_image
            )
            response_ids = service_output['completions']

            if TrainerTools().parallel.is_main_process and response_ids:
                gen_text = TrainerTools().tokenizer.decode(response_ids[0])
                with open(os.path.join(_get_log_dir(), 'gen.txt'), 'a') as f:
                    f.write(f"{tag}, gen->{eval_prompt}{gen_text}\n")
        else:
            with unwrap_model_for_generation(self.train_model) as eval_model:
                if TrainerTools().parallel.is_main_process:
                    eval_model = self._check_eval_model(eval_model)
                    eval_model.eval()

                    tokens = torch.tensor(TrainerTools().tokenizer.encode(eval_prompt), dtype=torch.long).unsqueeze(0)
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
        global_steps_since_last_save = 0
        global_steps_since_last_eval = 0

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
                                pixel_values=pixel_values,
                                return_logits=self.return_logits
                            )

                            # calc loss
                            loss, ce_loss = self._calc_loss(inputs, attention_mask, result, labels)
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
                            global_steps_since_last_save += 1
                            global_steps_since_last_eval += 1

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

                            if 0 < self.train_config.save_interval <= global_steps_since_last_save:
                                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                                save_steps(
                                    epoch=epoch,
                                    file_idx=file_idx,
                                    batch_idx=batch + 1,
                                    lr_scheduler=self.lr_scheduler
                                )
                                global_steps_since_last_save = 0

                            if 0 < self.train_config.eval_interval <= global_steps_since_last_eval:
                                self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')
                                global_steps_since_last_eval = 0
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
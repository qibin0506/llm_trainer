from typing import Tuple, List, Optional, Dict, Any
from contextlib import nullcontext
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .tools import TrainerTools
from .log import Logger
from .train_configs import TrainConfig, RewardFun, GenerationService
from .utils import (
    autocast,
    calc_position_ids,
    _build_deepspeed_kwargs,
    _build_data_loader_config
)
from .scheduler import LRScheduler, WarmupCosineAnnealingLRScheduler, NoneLRScheduler
from .partition_utils import zero3_trace_guard
from .muon import MuonWithAdamW, _get_ds_muon
from .ppo_trainer import PPOTrainer, ValueModel


class PPOValueWarmupTrainer(PPOTrainer):
    """
    PPOValueWarmupTrainer
    继承自 PPOTrainer，专门用于在 PPO 联合强化学习训练前对 Value (Critic) 价值模型进行独立预热的 Trainer。

    Args:
        train_config:
            - 全局训练配置，必须包含 ppo_config。
        reward_func:
            - 奖励打分函数，支持 1D 标量或 2D 逐 Token 稠密序列。
        generation_service:
            - 外部自定义生成服务接口（可选）
        max_warmup_steps:
            - 最多预热的全局优化更新步数。若达到此步数则提前终止预热（默认 None，表示跑完 n_epochs）。
        bootstrap_truncated:
            - 是否对未正常输出 EOS 的截断样本进行价值 Bootstrap。默认 False，即采用无偏的纯蒙特卡洛折现回报（Reward-to-Go），
              彻底阻断未初始化的随机 Critic 产生噪声回传；设为 True 时将使用安全截断的 last_values 进行 Bootstrap。
    """
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            reward_func: RewardFun,
            generation_service: Optional[GenerationService] = None,
            max_warmup_steps: Optional[int] = None,
            bootstrap_truncated: bool = False
    ):
        self.max_warmup_steps = max_warmup_steps
        self.bootstrap_truncated = bootstrap_truncated
        super().__init__(
            train_config=train_config,
            reward_func=reward_func,
            generation_service=generation_service,
            eval_prompts=[]
        )
        self.policy_model = self._init_policy_model()

    def _init_ref_model(self):
        # 预热阶段仅拟合价值函数，策略不更新，完全不需要 Reference 模型计算 KL 惩罚，
        # 返回 None 可规避未配 ref_model_weights_path 时的异常，并节省整整一份模型的显存
        return None

    @property
    def total_steps(self) -> int:
        return getattr(self.lr_scheduler, 'cur_steps', 0)

    def _init_policy_model(self) -> nn.Module:
        return self._init_frozen_model(self.train_config.init_weights_path)

    def _eval(self, tag: str): ...

    def _check_eval_model(self, eval_model):
        return self.policy_model

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs: Optional[Dict[str, Any]] = None
        ds_config = self.train_config.ds_config

        if self.is_ds and ds_config:
            value_optim_config = self.ppo_config.value_optim_config or self.train_config.optim_config
            parallel_kwargs = _build_deepspeed_kwargs(
                ds_config,
                self.gradient_accumulation_steps,
                self.ppo_config.ppo_batch_size,
                value_optim_config.optim_type,
                gradient_clipping=self.train_config.gradient_clipping
            )

        data_loader_kwargs, sampler_kwargs = _build_data_loader_config(
            self.train_config.data_loader_config,
            self.train_config.batch_size
        )

        data_loader_kwargs.update({"collate_fn": lambda x: x})
        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _init_train_model_and_optim(self, initial_lr: float):
        with self._new_model_context(self.parallel_kwargs):
            value_model = ValueModel(
                self._new_model(self.train_config),
                freeze_backbone=self.ppo_config.freeze_value_backbone
            )

        if self.train_config.is_gradient_checkpointing and not self.ppo_config.freeze_value_backbone:
            if self.is_ds:
                import deepspeed
                value_model.gradient_checkpointing_enable(checkpoint_func=deepspeed.checkpointing.checkpoint)
            else:
                value_model.gradient_checkpointing_enable()

        if TrainerTools().parallel.is_main_process:
            total_params = sum(getattr(p, 'ds_numel', p.numel()) for p in value_model.parameters())
            trainable_params = sum(getattr(p, 'ds_numel', p.numel()) for p in value_model.parameters() if p.requires_grad)
            Logger.std_log(f"[ValueModel] Total params: {total_params:,}, Trainable: {trainable_params:,} ({total_params * 4 / (1024 * 1024):.2f} MB)")

        if self.train_config.init_weights_path is not None:
            self._load_external_weights(
                value_model,
                self.train_config.init_weights_path,
                prefixes=['base_model.', '']
            )

        if self.ppo_config.value_model_weights_path is not None:
            self._load_external_weights(
                value_model,
                self.ppo_config.value_model_weights_path,
                prefixes=['', 'base_model.', 'value_model.'],
                exclude_prefixes=['policy_model.']
            )

        optimizer = self._config_optim(value_model, initial_lr)

        model, optim = TrainerTools().parallel.process(
            model=value_model,
            optimizer=optimizer,
            kwargs=self.parallel_kwargs
        )

        return model, optim

    def _get_optim_cls(self, optim_config=None):
        if optim_config is None:
            optim_config = self.ppo_config.value_optim_config if self.ppo_config.value_optim_config else self.train_config.optim_config
        return super()._get_optim_cls(optim_config)

    def _config_optim(self, model, initial_lr):
        value_config = self.ppo_config.value_optim_config if self.ppo_config.value_optim_config else self.train_config.optim_config
        optim_cls = self._get_optim_cls(value_config)
        param_data = self._build_optimizer_param_groups(model, value_config, name_prefix="value")

        if param_data["type"] == "muon":
            muon_groups = param_data["muon_groups"]
            adamw_groups = param_data["adamw_groups"]
            all_groups = muon_groups + adamw_groups
            muon_wd = value_config.muon_weight_decay
            if muon_wd is None:
                muon_wd = value_config.weight_decay if value_config.weight_decay is not None else 0.01

            if self.is_ds:
                unified_momentum = value_config.muon_momentum if value_config.muon_momentum is not None else 0.95
                for group in muon_groups:
                    group["momentum"] = unified_momentum

                ds_muon = _get_ds_muon(all_groups)
                if ds_muon is None:
                    raise RuntimeError('Current DeepSpeed version does not support Muon optimizer.')
                return ds_muon

            return MuonWithAdamW(
                muon_params=muon_groups,
                adamw_params=adamw_groups,
                muon_lr=value_config.muon_lr if value_config.muon_lr is not None else 0.02,
                muon_momentum=value_config.muon_momentum if value_config.muon_momentum is not None else 0.95,
                muon_weight_decay=muon_wd,
                muon_ns_steps=value_config.muon_ns_steps if value_config.muon_ns_steps is not None else 5,
                muon_adjust_lr_fn=value_config.muon_adjust_lr_fn,
                adamw_lr=value_config.initial_lr,
                adamw_betas=value_config.betas if value_config.betas is not None else (0.9, 0.95),
                adamw_weight_decay=value_config.weight_decay if value_config.weight_decay is not None else 0.01,
                adamw_cls=optim_cls
            )
        else:
            all_groups = param_data["groups"]
            default_betas = value_config.betas if value_config.betas else ((0.95, 0.98) if value_config.optim_type == 'lion' else (0.9, 0.999))
            default_weight_decay = value_config.weight_decay if value_config.weight_decay is not None else (0.015 if value_config.optim_type == 'lion' else 0.01)

            return optim_cls(
                all_groups,
                lr=value_config.initial_lr,
                betas=default_betas,
                weight_decay=default_weight_decay
            )

    def _init_lr_scheduler(self, initial_lr: float, optimizer) -> LRScheduler:
        value_config = self.ppo_config.value_optim_config if self.ppo_config.value_optim_config else self.train_config.optim_config
        real_initial_lr = value_config.initial_lr

        param_groups = getattr(optimizer, 'param_groups', None)
        if param_groups is None and hasattr(optimizer, 'optimizer'):
            param_groups = getattr(optimizer.optimizer, 'param_groups', None)
        if param_groups:
            for g in param_groups:
                if g.get('max_lr') is None:
                    params = g.get('params', [])
                    if g.get('use_muon', False) or any(getattr(p, 'use_muon', False) for p in params):
                        g['max_lr'] = value_config.muon_lr if value_config.muon_lr is not None else 0.02
                    else:
                        g['max_lr'] = value_config.max_lr if value_config.max_lr is not None else value_config.initial_lr

        if value_config.enable_lr_scheduler:
            warmup_iters = value_config.warmup_iters
            min_lr = value_config.min_lr
            max_lr = value_config.max_lr
            cosine_annealing_period = value_config.cosine_annealing_period
            cosine_annealing_period_mul = value_config.cosine_annealing_period_mul
            return WarmupCosineAnnealingLRScheduler(
                optimizer=optimizer,
                warmup_iters=warmup_iters,
                initial_lr=real_initial_lr,
                min_lr=min_lr,
                max_lr=max_lr,
                cosine_annealing_period=cosine_annealing_period,
                cosine_annealing_period_mul=cosine_annealing_period_mul,
                need_log=TrainerTools().parallel.is_main_process
            )
        else:
            return NoneLRScheduler(real_initial_lr)

    def _get_rollout_model_for_generation(self):
        return self.policy_model

    def _get_rollout_generation_model(self, unwrapped_model):
        return unwrapped_model

    def _compute_values(
            self,
            model: nn.Module,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            batch_size: Optional[int] = None
    ) -> torch.Tensor:
        is_no_grad = not torch.is_grad_enabled()

        def _step_forward(inp_ids, att_mask):
            guard = zero3_trace_guard(model) if is_no_grad else nullcontext()
            with guard:
                pos_ids = calc_position_ids(att_mask)
                values, _ = model(inp_ids, attention_mask=att_mask, position_ids=pos_ids)
                return values

        if batch_size is None or batch_size <= 0 or batch_size >= input_ids.size(0):
            return _step_forward(input_ids, attention_mask)

        all_values = []
        for i in range(0, input_ids.size(0), batch_size):
            c_input_ids = input_ids[i: i + batch_size]
            c_attention_mask = attention_mask[i: i + batch_size] if attention_mask is not None else None
            values = _step_forward(c_input_ids, c_attention_mask)
            all_values.append(values)

        return torch.cat(all_values, dim=0)

    def _evaluate_rollout_model(
            self,
            full_ids: torch.Tensor,
            full_attention_mask: torch.Tensor,
            completion_ids: torch.Tensor,
            loss_mask: torch.Tensor,
            can_skip_policy: bool
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        eval_batch_size = self.rollout_eval_batch_size
        values = self._compute_values(
            self.train_model,
            full_ids,
            full_attention_mask,
            batch_size=eval_batch_size
        )
        # 预热阶段策略模型冻结，无需计算新旧策略 log_probs 与参考模型 log_probs，彻底省去两次无谓的完整 LLM 前向开销
        return None, values, None

    def _compute_elementwise_loss(self, pred_values: torch.Tensor, target_returns: torch.Tensor) -> torch.Tensor:
        pred_values = pred_values.float()
        target_returns = target_returns.float()
        if self.ppo_config.value_loss_type == 'huber':
            return F.smooth_l1_loss(pred_values, target_returns, reduction='none', beta=self.ppo_config.huber_delta)
        return 0.5 * ((pred_values - target_returns) ** 2)

    def _learning_phase(self, rollout_data: dict) -> dict:
        prompt_ids: torch.Tensor = rollout_data['prompt_ids']
        completion_ids: torch.Tensor = rollout_data['completion_ids']
        old_values: torch.Tensor = rollout_data['values']
        rewards: torch.Tensor = rollout_data['rewards']
        dones: torch.Tensor = rollout_data['dones']

        loss_mask = rollout_data['loss_mask']
        completion_pad_mask = rollout_data['completion_pad_mask']

        batch_size = prompt_ids.shape[0]
        comp_len = completion_ids.shape[1]

        comp_values = old_values[:, -(comp_len + 1):]
        next_state_values = comp_values[:, 1:]

        last_token_indices = rollout_data['last_token_indices'].clamp(min=0)
        valid_indices_mask = rollout_data['valid_indices_mask']
        batch_range = rollout_data['batch_range']
        last_values = torch.zeros(batch_size, device=prompt_ids.device, dtype=old_values.dtype)

        if valid_indices_mask.any():
            last_values[valid_indices_mask] = next_state_values[
                batch_range[valid_indices_mask], last_token_indices[valid_indices_mask]
            ]

        # 核心：预热阶段采用无偏差的纯 Monte Carlo 折现累积回报（Reward-to-Go）
        # 彻底消除未初始化的随机 Critic (last_values) 对目标回报的污染与漂移
        gamma = self.ppo_config.gamma
        if self.bootstrap_truncated:
            # 若显式开启截断样本 bootstrap，采用安全截断保护防止随机初值剧烈扰动
            clamped_last_values = last_values.to(dtype=torch.float32).clamp(min=-10.0, max=10.0)
            bootstrap_ret = torch.where(
                dones.bool(),
                torch.zeros_like(clamped_last_values),
                clamped_last_values
            )
        else:
            # 默认：纯粹且无偏的 Monte Carlo 回报，未完成截断样本 bootstrap 设为 0，彻底切断随机 Critic 的噪声注入
            bootstrap_ret = torch.zeros(batch_size, device=rewards.device, dtype=torch.float32)

        next_ret = bootstrap_ret
        has_next_act = torch.zeros(batch_size, device=rewards.device, dtype=torch.bool)
        returns = torch.zeros_like(rewards)

        for t in reversed(range(comp_len)):
            is_act = loss_mask[:, t]
            next_val = torch.where(has_next_act, next_ret, bootstrap_ret)
            ret = rewards[:, t] + gamma * next_val

            returns[:, t] = torch.where(is_act, ret, 0.0)
            next_ret = torch.where(is_act, ret, next_ret)
            has_next_act = has_next_act | is_act

        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_masks = self._calc_attention_mask(prompt_ids)
        attention_mask = torch.cat((prompt_masks, completion_pad_mask), dim=1)

        warmup_stats = {
            "loss": 0.0, "value_loss": 0.0, "value_mean": 0.0,
            "return_mean": 0.0, "value_error": 0.0, "completion_len": 0.0
        }

        ppo_batch_size = self.ppo_config.ppo_batch_size
        total_micro_batches_processed = 0

        for _ in range(self.ppo_config.ppo_epochs):
            indices = torch.randperm(batch_size, device=TrainerTools().parallel.device)
            micro_batch_idx = 0

            for i in range(0, batch_size, ppo_batch_size):
                mini_batch_indices = indices[i: i + ppo_batch_size]

                mb_input_ids = input_ids[mini_batch_indices]
                mb_attention_mask = attention_mask[mini_batch_indices]
                mb_loss_mask = loss_mask[mini_batch_indices]
                mb_returns = returns[mini_batch_indices]
                mb_position_ids = calc_position_ids(mb_attention_mask)
                mb_comp_len = completion_ids.size(1)

                with autocast(TrainerTools().parallel.device_type):
                    current_values_full, value_aux_loss = self.train_model(
                        mb_input_ids,
                        attention_mask=mb_attention_mask,
                        position_ids=mb_position_ids
                    )
                    current_values = current_values_full[:, -(mb_comp_len + 1): -1]

                    # 预热阶段不使用 value clipping（防止截断从 0 到真实奖励尺度的更新梯度）
                    vf_loss = self._compute_elementwise_loss(current_values, mb_returns)
                    value_loss = (vf_loss * mb_loss_mask).sum() / mb_loss_mask.sum().clamp(min=1.0)
                    # 预热阶段无需乘 vf_coef，保持优化器真实学习率步长
                    total_loss = value_loss
                    if value_aux_loss is not None:
                        total_loss = total_loss + value_aux_loss.to(total_loss.dtype)

                with torch.no_grad():
                    value_mean = (current_values * mb_loss_mask).sum() / mb_loss_mask.sum().clamp(min=1.0)
                    return_mean = (mb_returns * mb_loss_mask).sum() / mb_loss_mask.sum().clamp(min=1.0)
                    value_error = value_mean - return_mean
                    completion_len = mb_loss_mask.sum(dim=-1).float().mean()

                is_last_mini_batch = (i + ppo_batch_size >= batch_size)
                total_mbs = math.ceil(batch_size / ppo_batch_size)
                mb_idx_0 = i // ppo_batch_size
                window_start_mb = (mb_idx_0 // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
                actual_accum_steps = min(self.gradient_accumulation_steps, total_mbs - window_start_mb)

                if self.is_ds:
                    if is_last_mini_batch:
                        self.train_model.set_gradient_accumulation_boundary(True)
                        need_update_step = True
                    else:
                        need_update_step = self.train_model.is_gradient_accumulation_boundary()

                    if actual_accum_steps < self.gradient_accumulation_steps:
                        scale_factor = self.gradient_accumulation_steps / actual_accum_steps
                        self._backward_loss(total_loss * scale_factor, actual_accum_steps)
                    else:
                        self._backward_loss(total_loss, self.gradient_accumulation_steps)
                else:
                    micro_batch_idx += 1
                    need_update_step = (
                            micro_batch_idx % self.gradient_accumulation_steps == 0
                            or is_last_mini_batch
                    )
                    self._backward_loss(total_loss, actual_accum_steps)

                warmup_stats["loss"] += total_loss.detach().item()
                warmup_stats["value_loss"] += value_loss.detach().item()
                warmup_stats["value_mean"] += value_mean.detach().item()
                warmup_stats["return_mean"] += return_mean.detach().item()
                warmup_stats["value_error"] += value_error.detach().item()
                warmup_stats["completion_len"] += completion_len.detach().item()
                total_micro_batches_processed += 1

                if need_update_step:
                    self._update_step(is_last_step=is_last_mini_batch)

        if total_micro_batches_processed > 0:
            for key in warmup_stats:
                warmup_stats[key] /= total_micro_batches_processed

        return warmup_stats

    def _log_train_stats(
            self,
            train_stats: dict,
            rollout_data: dict,
            epoch: int,
            file_idx: int,
            file_count: int,
            batch: int,
            batch_count_per_file: int
    ):
        stats_tensor = torch.tensor([
            train_stats['loss'],
            train_stats['value_loss'],
            train_stats['value_mean'],
            train_stats['return_mean'],
            train_stats['value_error'],
            rollout_data['env_rewards'].item(),
            train_stats['completion_len'],
        ], device=TrainerTools().parallel.device)

        if TrainerTools().parallel.parallel_train:
            if TrainerTools().parallel.device_type == 'mlu':
                dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
                stats_tensor.div_(TrainerTools().parallel.world_size)
            else:
                dist.all_reduce(stats_tensor, dist.ReduceOp.AVG)

        cur_val_lr = self.lr_scheduler.cur_lr
        log_keys = {
            'epoch': epoch,
            'file': f'{file_idx + 1}/{file_count}',
            'batch': f'{batch + 1}/{batch_count_per_file}',
            'step': f'{self.total_steps}' + (f'/{self.max_warmup_steps}' if self.max_warmup_steps else '')
        }

        self._log(
            keys=log_keys,
            values={
                'loss/total': stats_tensor[0].item(),
                'loss/value': stats_tensor[1].item(),
                'value/mean': stats_tensor[2].item(),
                'value/return': stats_tensor[3].item(),
                'value/error': stats_tensor[4].item(),
                'env/reward_total': stats_tensor[5].item(),
                'env/completion_len': stats_tensor[6].item(),
                'lr/value': cur_val_lr
            }
        )

    def _should_stop_training(self) -> bool:
        local_stop = (self.max_warmup_steps is not None and self.total_steps >= self.max_warmup_steps)
        if TrainerTools().parallel.parallel_train and dist.is_initialized():
            stop_tensor = torch.tensor(1 if local_stop else 0, dtype=torch.int32, device=TrainerTools().parallel.device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            stop = bool(stop_tensor.item() > 0)
        else:
            stop = local_stop

        if stop and TrainerTools().parallel.is_main_process:
            Logger.std_log(
                f"[PPOValueWarmupTrainer] Reached target warmup steps ({self.total_steps}/{self.max_warmup_steps}). "
                f"Value warmup target achieved!"
            )
        return stop

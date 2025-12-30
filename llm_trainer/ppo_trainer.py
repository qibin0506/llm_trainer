from typing import Tuple, List, Union, Callable, Optional
import gc
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import torch.nn as nn

from llm_model import LlmModel, VlmModel

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig
from .dataset import RLDataset
from .loss import PPOLoss
from .tools import TrainerTools
from .generate_utils import batch_generate
from .utils import (
    autocast,
    left_pad_sequence,
    log_softmax,
    masked_whiten,
    disable_dropout_in_model,
    calc_position_ids
)
from .partition_utils import unwrap_model_for_generation
from .log import Logger
from .checkpoint import (
    save_checkpoint,
    save_steps,
)


class ValueModel(nn.Module):
    def __init__(self, base_model: Union[LlmModel, VlmModel]):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1, bias=True)
        self.value_head.weight.data.normal_(mean=0.0, std=0.01)
        self.value_head.bias.data.zero_()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        outputs = self.base_model(*args, **kwargs)
        # [batch_size, seq_len, hidden_size]
        last_hidden_state = outputs['hidden_states']
        # [batch_size, seq_len, 1]
        values = self.value_head(last_hidden_state)
        # [batch_size, seq_len]
        return values.squeeze(-1)


class PolicyAndValueModelWrapper(nn.Module):
    def __init__(self, policy_model: nn.Module, value_model: nn.Module):
        super().__init__()
        self.policy_model = policy_model
        self.value_model = value_model

    def forward(self, *args, **kwargs):
        return self.policy_model(*args, **kwargs), self.value_model(*args, **kwargs)


class PPOTrainer(BaseTrainer):
    """
    reward_func(prompt_ids, complete_ids, answer_ids) -> scores
    """

    def __init__(
            self,
            *,
            train_config: TrainConfig,
            reward_func: Callable[[List[torch.Tensor], torch.Tensor, List[Optional[torch.Tensor]]], List[float]],
            eval_prompts: List[str]
    ):
        self.ppo_config = train_config.ppo_config

        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            gradient_accumulation_steps=self.ppo_config.gradient_accumulation_steps
        )
        self.reward_func = reward_func

        self.ref_model = self._init_ref_model()

        if self.train_config.ppo_config.normalize_rewards and self.train_config.ppo_config.whiten_rewards:
            self.train_config.ppo_config.whiten_rewards = False
            if TrainerTools().parallel.is_main_process:
                Logger.std_log('WARN: ppo_config.normalize_rewards is enabled, ppo_config.whiten_rewards must be disabled.')

    def _init_train_model_and_optim(self, initial_lr: float):
        policy_model = self._new_model(self.train_config)
        value_model = ValueModel(self._new_model(self.train_config))
        train_model = PolicyAndValueModelWrapper(policy_model, value_model)

        if self.train_config.init_state_dict:
            policy_model.load_state_dict(self.train_config.init_state_dict)
            value_model.base_model.load_state_dict(self.train_config.init_state_dict)
            self.train_config.init_state_dict = None

        if self.train_config.ppo_config.value_model_checkpoint:
            value_model.load_state_dict(self.train_config.ppo_config.value_model_checkpoint)
            self.train_config.ppo_config.value_model_checkpoint = {}

        if TrainerTools().parallel.is_main_process:
            for name, model in zip(['policy', 'value'], [policy_model, value_model]):
                total_params = sum(p.numel() for p in model.parameters())
                Logger.std_log(f"Total number of {name} model parameters: {total_params:,}")

                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                Logger.std_log(f"Trainable number of {name} model parameters: {trainable_params:,}")

                total_size_bytes = total_params * 4
                total_size_mb = total_size_bytes / (1024 * 1024)
                Logger.std_log(f"Total size of {name} model model: {total_size_mb:.2f} MB")

        model, optim = TrainerTools().parallel.process(
            model=train_model,
            optimizer=self._config_optim(train_model, initial_lr),
            kwargs=self.parallel_kwargs
        )

        return model, optim

    def _init_ref_model(self):
        ref_model = self._new_model(self.train_config)

        if self.train_config.ppo_config.ref_model_checkpoint:
            ref_model.load_state_dict(self.train_config.ppo_config.ref_model_checkpoint)
            self.train_config.ppo_config.ref_model_checkpoint = {}

        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        ref_model, _ = TrainerTools().parallel.process(
            model=ref_model,
            optimizer=None,
            kwargs=self._init_ref_model_args(),
            save_instance=False
        )

        return ref_model

    def _new_model(self, train_config: TrainConfig):
        model = super()._new_model(train_config)
        disable_dropout_in_model(model)
        return model

    def _init_loss(self):
        ppo_config = self.train_config.ppo_config
        criterion = PPOLoss(
            clip_eps=ppo_config.clip_eps,
            vf_coef=ppo_config.vf_coef
        )
        return criterion, None

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": lambda x: x})
        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        return RLDataset(file_path), file_path

    def _calc_loss(self, inputs, attention_mask, logits, labels): ...

    def _check_eval_model(self, eval_model):
        return eval_model.policy_model

    def _compute_advantages_and_returns(
            self,
            rewards: torch.Tensor,
            values: torch.Tensor,
            last_values: torch.Tensor,
            completion_mask: torch.Tensor,
            dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, lam = self.train_config.ppo_config.gamma, self.train_config.ppo_config.lam
        advantages_reversed = []
        last_gae_lam = 0
        seq_len = rewards.size(1)

        values = values * completion_mask
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_values = torch.where(dones, 0.0, last_values)
            else:
                next_values = values[:, t + 1]

            delta = rewards[:, t] + gamma * next_values - values[:, t]
            last_gae_lam = delta + gamma * lam * last_gae_lam * completion_mask[:, t]
            advantages_reversed.append(last_gae_lam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values

        return advantages * completion_mask, returns * completion_mask

    def _generate_rollout_data(self, batch_data: List[dict]) -> dict:
        ppo_config = self.train_config.ppo_config
        device = TrainerTools().parallel.device
        pad_token_id = TrainerTools().tokenizer.pad
        eos_token_id = TrainerTools().tokenizer.end

        prompts = [item["prompt"] for item in batch_data]
        answers = [item["answer"] for item in batch_data]

        prompt_ids = left_pad_sequence(prompts, padding_value=pad_token_id)
        prompt_ids = prompt_ids.to(device)
        prompt_masks = (prompt_ids != pad_token_id)
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            with unwrap_model_for_generation(self.train_model) as unwrapped_model:
                full_ids, logitss = batch_generate(
                    model=unwrapped_model.policy_model,
                    tokens=prompt_ids,
                    attention_mask=prompt_masks,
                    max_new_tokens=ppo_config.gen_max_new_tokens,
                    temperature=ppo_config.gen_temperature,
                    k=ppo_config.gen_k,
                    p=ppo_config.gen_p,
                    suppress_tokens=ppo_config.gen_suppress_tokens,
                    device=device
                )
                completion_ids = full_ids[:, prompt_len:]
                full_attention_mask = (full_ids != pad_token_id)
                full_position_ids = calc_position_ids(full_attention_mask)

                with autocast(TrainerTools().parallel.device_type):
                    value_output = unwrapped_model.value_model(
                        full_ids,
                        attention_mask=full_attention_mask,
                        position_ids=full_position_ids
                    )

            old_log_probs = log_softmax(logitss.float(), completion_ids)

            with unwrap_model_for_generation(self.ref_model) as unwrapped_ref_model:
                ref_outputs = unwrapped_ref_model(
                    full_ids,
                    attention_mask=full_attention_mask,
                    position_ids=full_position_ids
                )
                ref_logits_full = ref_outputs['logits']

            ref_logits_completion = ref_logits_full[:, prompt_len - 1: -1]
            ref_log_probs_completion = log_softmax(ref_logits_completion.float(), completion_ids)

            dones = torch.any(completion_ids == eos_token_id, dim=1)
            rewards = torch.zeros_like(completion_ids, dtype=torch.float32, device=device)
            completion_mask = (completion_ids != pad_token_id)

            if ppo_config.kl_beta > 0.0:
                logr = ref_log_probs_completion - old_log_probs
                kl = -logr if ppo_config.kl_estimator == "k1" else (logr.exp() - 1) - logr
                kl_rewards = -ppo_config.kl_beta * kl
                rewards += kl_rewards * completion_mask

            env_rewards_tensor = torch.tensor(
                self.reward_func(prompts, completion_ids, answers),
                dtype=torch.float32,
                device=device
            )

            if ppo_config.missing_eos_penalty is not None:
                env_rewards_tensor[~dones] -= ppo_config.missing_eos_penalty

            raw_reward_mean = env_rewards_tensor.mean()
            if self.train_config.ppo_config.normalize_rewards:
                batch_std = env_rewards_tensor.std()
                if torch.isnan(batch_std) or batch_std < 1e-8:
                    batch_std = 1.0

                env_rewards_tensor = (env_rewards_tensor - raw_reward_mean) / batch_std

            last_token_indices = completion_mask.sum(dim=1) - 1
            valid_indices_mask = last_token_indices >= 0

            if valid_indices_mask.any():
                valid_batch_indices = torch.arange(prompt_ids.size(0), device=device)[valid_indices_mask]
                valid_last_token_indices = last_token_indices[valid_indices_mask]
                valid_env_rewards = env_rewards_tensor[valid_indices_mask]
                rewards[valid_batch_indices, valid_last_token_indices] += valid_env_rewards

        return {
            'prompt_ids': prompt_ids.detach(),
            'completion_ids': completion_ids.detach(),
            'old_log_probs': old_log_probs.detach(),
            'values': value_output.detach(),
            'rewards': rewards.detach(),
            'env_rewards': raw_reward_mean.detach(),
            'dones': dones.detach(),
        }

    def _ppo_learning_phase(self, rollout_data: dict):
        ppo_config = self.train_config.ppo_config

        prompt_ids: torch.Tensor = rollout_data['prompt_ids']
        completion_ids: torch.Tensor = rollout_data['completion_ids']
        old_log_probs: torch.Tensor = rollout_data['old_log_probs']
        old_values: torch.Tensor = rollout_data['values']
        rewards: torch.Tensor = rollout_data['rewards']
        dones: torch.Tensor = rollout_data['dones']

        prompt_len = prompt_ids.shape[1]
        batch_size = prompt_ids.shape[0]

        values_for_gae = old_values[:, prompt_len - 1: -1]
        last_values = old_values[:, -1]
        assert values_for_gae.shape[1] == completion_ids.shape[1]

        completion_mask: torch.Tensor = (completion_ids != TrainerTools().tokenizer.pad)

        if ppo_config.whiten_rewards:
            rewards = masked_whiten(rewards, completion_mask, shift_mean=False)
            rewards = torch.masked_fill(rewards, ~completion_mask, 0.0)

        advantages, returns = self._compute_advantages_and_returns(
            rewards, values_for_gae, last_values, completion_mask, dones
        )

        advantages_whitened = masked_whiten(advantages, completion_mask, shift_mean=True)
        advantages_whitened = torch.masked_fill(advantages_whitened, ~completion_mask, 0.0)

        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = (input_ids != TrainerTools().tokenizer.pad)

        ppo_stats = {
            "loss": 0.0, "moe_aux_loss": 0.0, "actor_loss": 0.0,
            "value_loss": 0.0, "approx_kl": 0.0, "clip_frac": 0.0
        }

        grad_acc_steps = max(1, self.gradient_accumulation_steps)
        ppo_batch_size = ppo_config.ppo_batch_size
        num_micro_batches = (batch_size + ppo_batch_size - 1) // ppo_batch_size
        total_micro_batches_processed = 0

        for ppo_epoch in range(ppo_config.ppo_epochs):
            indices = torch.randperm(batch_size, device=TrainerTools().parallel.device)

            for i in range(0, batch_size, ppo_batch_size):
                mini_batch_indices = indices[i:i + ppo_batch_size]
                micro_batch_idx = i // ppo_batch_size
                is_last_micro_batch = (micro_batch_idx == num_micro_batches - 1)
                need_update_grad = ((micro_batch_idx + 1) % grad_acc_steps == 0) or is_last_micro_batch

                if is_last_micro_batch:
                    remainder = (micro_batch_idx + 1) % grad_acc_steps
                    actual_acc_steps = remainder if remainder > 0 else grad_acc_steps
                else:
                    actual_acc_steps = grad_acc_steps

                if TrainerTools().parallel.parallel_train:
                    self.train_model.require_backward_grad_sync = need_update_grad

                mb_input_ids = input_ids[mini_batch_indices]
                mb_attention_mask = attention_mask[mini_batch_indices]
                mb_completion_ids = completion_ids[mini_batch_indices]
                mb_completion_mask = completion_mask[mini_batch_indices]
                mb_old_log_probs = old_log_probs[mini_batch_indices]
                mb_values = values_for_gae[mini_batch_indices]
                mb_returns = returns[mini_batch_indices]
                mb_advantages = advantages_whitened[mini_batch_indices]
                mb_position_ids = calc_position_ids(mb_attention_mask)

                with autocast(TrainerTools().parallel.device_type):
                    policy_output, value_output = self.train_model(
                        mb_input_ids,
                        attention_mask=mb_attention_mask,
                        position_ids=mb_position_ids
                    )

                    target_dtype = policy_output['logits'].dtype
                    mb_old_log_probs = mb_old_log_probs.to(target_dtype)
                    mb_values = mb_values.to(target_dtype)
                    mb_returns = mb_returns.to(target_dtype)
                    mb_advantages = mb_advantages.to(target_dtype)

                    logits_completion = policy_output['logits'][:, prompt_len - 1: -1]
                    current_log_probs = log_softmax(logits_completion, mb_completion_ids)
                    current_values = value_output[:, prompt_len - 1: -1]

                    loss, actor_loss, value_loss, approx_kl, clip_frac = self.criterion(
                        log_probs=current_log_probs,
                        old_log_probs=mb_old_log_probs,
                        values=current_values,
                        old_values=mb_values,
                        returns=mb_returns,
                        advantages=mb_advantages,
                        mask=mb_completion_mask
                    )

                    aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                    if policy_output.get('aux_loss') and self.train_config.loss_config.aux_loss_coef:
                        aux_loss = self.train_config.loss_config.aux_loss_coef * policy_output['aux_loss']

                total_loss = loss + aux_loss
                scaled_total_loss = total_loss / actual_acc_steps
                self._backward_loss(scaled_total_loss)

                ppo_stats["loss"] += total_loss.detach().item()
                ppo_stats["moe_aux_loss"] += aux_loss.detach().item()
                ppo_stats["actor_loss"] += actor_loss.detach().item()
                ppo_stats["value_loss"] += value_loss.detach().item()
                ppo_stats["approx_kl"] += approx_kl.detach().item()
                ppo_stats["clip_frac"] += clip_frac.detach().item()
                total_micro_batches_processed += 1

                if need_update_grad:
                    self._apply_grad_clipping()
                    self._apply_step()

        if total_micro_batches_processed > 0:
            for key in ppo_stats:
                ppo_stats[key] /= total_micro_batches_processed

        return ppo_stats

    def train(self):
        global_steps = 0
        skipping_train = False

        for epoch in range(self.train_config.n_epochs):
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

                    if skipping_train:
                        TrainerTools().parallel.wait('skip train')
                        skipping_train = False

                    rollout_data = self._generate_rollout_data(batch_data)
                    torch.cuda.empty_cache()

                    try:
                        ppo_stats = self._ppo_learning_phase(rollout_data)

                        stats_tensor = torch.tensor([
                            ppo_stats['loss'],
                            ppo_stats['moe_aux_loss'],
                            ppo_stats['actor_loss'],
                            ppo_stats['value_loss'],
                            ppo_stats['approx_kl'],
                            ppo_stats['clip_frac'],
                            rollout_data['env_rewards'].item()
                        ], device=TrainerTools().parallel.device)

                        if TrainerTools().parallel.parallel_train:
                            dist.all_reduce(stats_tensor, op=dist.ReduceOp.AVG)

                        ppo_stats['loss'] = stats_tensor[0].item()
                        ppo_stats['moe_aux_loss'] = stats_tensor[1].item()
                        ppo_stats['actor_loss'] = stats_tensor[2].item()
                        ppo_stats['value_loss'] = stats_tensor[3].item()
                        ppo_stats['approx_kl'] = stats_tensor[4].item()
                        ppo_stats['clip_frac'] = stats_tensor[5].item()
                        reward_value = stats_tensor[6].item()

                        self._log(
                            keys={
                                'epoch': epoch,
                                'file': f'{file_idx + 1}/{file_count}',
                                'batch': f'{batch}/{batch_count_per_file}'
                            },
                            values={
                                'loss': ppo_stats['loss'],
                                'moe_aux_loss': ppo_stats['moe_aux_loss'],
                                'actor_loss': ppo_stats['actor_loss'],
                                'value_loss': ppo_stats['value_loss'],
                                'approx_kl': ppo_stats['approx_kl'],
                                'clip_frac': ppo_stats['clip_frac'],
                                'rewards': reward_value
                            }
                        )
                    except Exception as e:
                        self._on_exception(e, epoch, batch)
                    finally:
                        save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                        if (batch - last_ckpt_batch) >= self.train_config.eval_config.eval_batch_interval:
                            save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                            last_ckpt_batch = batch
                            self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')

                        torch.cuda.empty_cache()

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
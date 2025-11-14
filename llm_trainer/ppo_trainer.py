from typing import Tuple, List, Union, Callable, Optional
import torch
from torch.utils.data import Dataset
import torch.nn as nn

from llm_model import LlmModel, VlmModel

from .trainer import Trainer
from .train_configs import TrainConfig
from .dataset import RLDataset
from .loss import PPOLoss
from .tools import TrainerTools
from .generate_utils import batch_generate
from .utils import (
    autocast,
    left_pad_sequence,
    log_softmax,
    masked_whiten
)
from .partition_utils import unwrap_model_for_generation
from .log import log
from .checkpoint import (
    save_checkpoint,
    save_steps,
)


class ValueModel(nn.Module):
    def __init__(self, base_model: Union[LlmModel, VlmModel]):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)
        self.value_head.weight.data.zero_()

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


class PPOTrainer(Trainer):
    """
    reward_func(prompt_ids, complete_ids, answer_ids) -> scores
    """

    def __init__(
            self,
            *,
            train_config: TrainConfig,
            reward_func: Callable[[List[torch.Tensor], torch.Tensor, List[Optional[torch.Tensor]]], List[float]],
            eval_prompts: List[str],
            eval_image_tags: Optional[List[str]] = None
    ):
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            eval_image_tags=eval_image_tags
        )
        self.packed_sequences = False
        self.reward_func = reward_func

        self.ref_model = self._init_ref_model()

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
                log(f"Total number of {name} model parameters: {total_params:,}")

                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                log(f"Trainable number of {name} model parameters: {trainable_params:,}")

                total_size_bytes = total_params * 4
                total_size_mb = total_size_bytes / (1024 * 1024)
                log(f"Total size of {name} model model: {total_size_mb:.2f} MB")

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

        ref_model, _ = TrainerTools().parallel.process(
            model=ref_model,
            optimizer=None,
            kwargs=self._init_ref_model_args(),
            save_instance=False
        )

        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        return ref_model

    def _init_loss(self):
        ppo_config = self.train_config.ppo_config
        criterion = PPOLoss(clip_eps=ppo_config.clip_eps, vf_coef=ppo_config.vf_coef)
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

    def _compute_advantages_and_returns(self, rewards: torch.Tensor, values: torch.Tensor,
                                        completion_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma, lam = self.train_config.ppo_config.gamma, self.train_config.ppo_config.lam
        last_gae_lam = 0
        advantages_reversed = []
        seq_len = rewards.size(1)

        for t in reversed(range(seq_len)):
            next_values = values[:, t + 1] if t < seq_len - 1 else 0.0
            delta = rewards[:, t] + gamma * next_values - values[:, t]
            last_gae_lam = delta + gamma * lam * last_gae_lam
            advantages_reversed.append(last_gae_lam)

        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values

        return advantages * completion_mask, returns * completion_mask

    def _generate_rollout_data(self, batch_data: List[dict]) -> dict:
        """
        生成 rollout 数据
        """
        ppo_config = self.train_config.ppo_config
        device = TrainerTools().parallel.device
        pad_token_id = TrainerTools().tokenizer.pad

        prompts = [item["prompt"].to(device) for item in batch_data]
        answers = [item["answer"] for item in batch_data]

        prompt_ids = left_pad_sequence(prompts, padding_value=pad_token_id)
        prompt_masks = (prompt_ids != pad_token_id)
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            with unwrap_model_for_generation(self.train_model) as unwrapped_model:
                full_ids, logits = batch_generate(
                    model=unwrapped_model.policy_model,
                    tokens=prompt_ids,
                    pad_token_id=pad_token_id,
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
                completion_mask = full_attention_mask[:, prompt_len:]

                assert logits.shape[1] == completion_ids.shape[1], "Logits and completion length mismatch!"
                old_log_probs = log_softmax(logits, completion_ids)

                with autocast(device):
                    value_output = unwrapped_model.value_model(full_ids, attention_mask=full_attention_mask)

            ref_logits = self.ref_model(full_ids, attention_mask=full_attention_mask)['logits']

            ref_logits_completion = ref_logits[:, prompt_len - 1: -1]
            ref_log_probs_completion = log_softmax(ref_logits_completion, completion_ids)

            rewards = torch.zeros_like(completion_ids, dtype=logits.dtype, device=device)

            # KL 散度奖励惩罚项
            if ppo_config.kl_beta > 0.0:
                logr = ref_log_probs_completion - old_log_probs
                kl = -logr if ppo_config.kl_estimator == "k1" else (logr.exp() - 1) - logr

                kl_rewards = -ppo_config.kl_beta * kl * completion_mask
                rewards += kl_rewards

            # 外部环境奖励（例如，来自奖励模型）
            env_rewards_tensor = torch.tensor(
                self.reward_func(prompts, completion_ids, answers),
                dtype=logits.dtype,
                device=device
            )

            if ppo_config.use_sparse_rewards:
                last_token_indices = completion_mask.sum(dim=1) - 1
                for i in range(prompt_ids.size(0)):
                    if last_token_indices[i] >= 0:
                        rewards[i, last_token_indices[i]] += env_rewards_tensor[i]
            else:
                # 密集奖励，奖励加到全部有效token上
                final_rewards = env_rewards_tensor.unsqueeze(-1)
                rewards += final_rewards * completion_mask

        return {
            'prompt_ids': prompt_ids.detach().clone(),
            'completion_ids': completion_ids.detach().clone(),
            'old_log_probs': old_log_probs.detach(),
            'values': value_output.detach(),
            'rewards': rewards.detach(),
            'env_rewards': env_rewards_tensor.detach(),
        }

    def _ppo_learning_phase(self, rollout_data: dict):
        """
        执行PPO学习阶段，根据 rollout 数据进行多轮优化。
        """
        prompt_ids = rollout_data['prompt_ids']
        completion_ids = rollout_data['completion_ids']
        old_log_probs = rollout_data['old_log_probs']
        old_values = rollout_data['values']
        rewards = rollout_data['rewards']

        prompt_len = prompt_ids.shape[1]
        values = old_values[:, prompt_len - 1: -1]
        completion_mask = (completion_ids != TrainerTools().tokenizer.pad)

        if self.train_config.ppo_config.whiten_rewards:
            rewards = masked_whiten(rewards, completion_mask, shift_mean=False)

        advantages, returns = self._compute_advantages_and_returns(rewards, values, completion_mask)
        advantages_whitened = masked_whiten(advantages, completion_mask, shift_mean=True)

        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = (input_ids != TrainerTools().tokenizer.pad)

        loss_with_aux_accumulation = 0
        loss_without_aux_accumulation = 0
        aux_loss_accumulation = 0
        actor_loss_accumulation = 0
        value_loss_accumulation = 0.0
        approx_kl_accumulation = 0.0
        clip_frac_accumulation = 0.0

        avg_epochs = self.train_config.ppo_config.ppo_epochs
        for _ in range(avg_epochs):
            with autocast(TrainerTools().parallel.device_type):
                policy_output, value_output = self.train_model(input_ids, attention_mask=attention_mask)

                logits_completion = policy_output['logits'][:, prompt_len - 1: -1]
                current_log_probs = log_softmax(logits_completion, completion_ids)
                current_values = value_output[:, prompt_len - 1: -1]

                loss, actor_loss, value_loss, approx_kl, clip_frac = self.criterion(
                    log_probs=current_log_probs,
                    old_log_probs=old_log_probs,
                    values=current_values,
                    old_values=values,
                    returns=returns,
                    advantages=advantages_whitened,
                    mask=completion_mask
                )

                aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
                if policy_output.get('aux_loss') and self.train_config.loss_config.aux_loss_coef:
                    aux_loss = self.train_config.loss_config.aux_loss_coef * policy_output['aux_loss']

            total_loss = loss + aux_loss
            self._backward_loss(total_loss)
            self._apply_grad_clipping()
            self._apply_step()

            loss_with_aux_accumulation += total_loss.detach().item()
            loss_without_aux_accumulation += loss.detach().item()
            aux_loss_accumulation += aux_loss.detach().item()
            actor_loss_accumulation += actor_loss.detach().item()
            value_loss_accumulation += value_loss.detach().item()
            approx_kl_accumulation += approx_kl.detach().item()
            clip_frac_accumulation += clip_frac.detach().item()

        return (
            loss_with_aux_accumulation / avg_epochs,
            loss_without_aux_accumulation / avg_epochs,
            aux_loss_accumulation / avg_epochs,
            actor_loss_accumulation / avg_epochs,
            value_loss_accumulation / avg_epochs,
            approx_kl_accumulation / avg_epochs,
            clip_frac_accumulation / avg_epochs
        )

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

                    try:
                        (avg_loss,
                         avg_loss_without_aux,
                         avg_aux_loss,
                         avg_actor_loss,
                         avg_value_loss,
                         avg_approx_kl,
                         avg_clip_frac) = self._ppo_learning_phase(rollout_data)

                        self._log(
                            keys={
                                'epoch': epoch,
                                'file': f'{file_idx + 1}/{file_count}',
                                'batch': f'{batch}/{batch_count_per_file}'
                            },
                            values={
                                'loss(with aux)': avg_loss,
                                'loss(without aux)': avg_loss_without_aux,
                                'aux_loss': avg_aux_loss,
                                'actor_loss': avg_actor_loss,
                                'value_loss': avg_value_loss,
                                'approx_kl': avg_approx_kl,
                                'clip_frac': avg_clip_frac,
                                'rewards': rollout_data['env_rewards'].mean().item()
                            }
                        )
                    except Exception as e:
                        self._on_exception(e, epoch, batch)
                    finally:
                        save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)

                        if (batch - last_ckpt_batch) >= self.train_config.eval_batch_interval:
                            save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                            last_ckpt_batch = batch
                            self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')

            if not skipping_train:
                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                save_checkpoint(model=self.train_model, optimizer=self.optimizer)

                TrainerTools().parallel.on_epoch_end(epoch)
                self._on_epoch_end(tag=f'epoch:{epoch}')

        TrainerTools().parallel.destroy()
from typing import Tuple, List, Callable, Optional
import gc
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig
from .dataset import RLDataset
from .loss import GRPOLoss
from .tools import TrainerTools
from .generate_utils import batch_generate
from .log import Logger
from .utils import (
    autocast,
    left_pad_sequence,
    log_softmax,
    disable_dropout_in_model,
    calc_position_ids
)

from .partition_utils import (
    sync_model_params,
    unwrap_model_for_generation
)

from .checkpoint import (
    save_checkpoint,
    save_steps,
)

class GRPOTrainer(BaseTrainer):
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
        self.grpo_config = train_config.grpo_config
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts
        )

        self.reward_func = reward_func
        self.ref_model = self._init_ref_model()

    def _init_ref_model(self):
        # beta == 0，不需要ref_model
        if self.grpo_config.loss_beta == 0.0:
            return None

        ref_model = self._new_model(self.train_config)

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
        criterion = GRPOLoss(
            beta=self.grpo_config.loss_beta,
            clip_eps_low=self.grpo_config.loss_clip_eps,
            clip_eps_high=self.grpo_config.loss_clip_eps_high,
            delta=self.grpo_config.loss_delta,
            importance_sampling_level=self.grpo_config.loss_importance_sampling_level,
            loss_type=self.grpo_config.loss_type,
            gen_max_new_tokens=self.grpo_config.gen_max_new_tokens
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

    def _compute_log_probs(
            self,
            model,
            input_ids,
            attention_mask
    ):
        position_ids = calc_position_ids(attention_mask)

        # [batch_size, total_seq_len, vocab_size]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids
        )

        # [batch_size, total_seq_len - 1, vocab_size]
        logits = outputs['logits'][:, :-1, :]
        input_ids = input_ids[:, 1:]

        # Compute and return the log probabilities for the selected tokens.
        return log_softmax(logits, input_ids), outputs['aux_loss']

    def _compute_group_relative_advantages(self, rewards):
        group_size = self.grpo_config.group_size

        # Reshape rewards to group by prompt
        # [batch, group_size]
        rewards_by_group = rewards.view(-1, group_size)

        # Compute mean and standard deviation for each prompt group
        # [batch]
        group_means = rewards_by_group.mean(dim=1)
        group_stds = rewards_by_group.std(dim=1)

        # Expand the means and stds to match the original flat rewards tensor shape
        # [batch*group_size]
        expanded_means = group_means.repeat_interleave(group_size)
        expanded_stds = group_stds.repeat_interleave(group_size)

        # Normalize rewards to get advantages
        # [batch*group_size]
        advantages = (rewards - expanded_means) / (expanded_stds + 1e-4)

        # [batch*group_size, 1]
        return advantages.unsqueeze(1)  # Add dimension for token-wise operations

    def _generate_completions(self, model, prompts, group_size: int):
        pad_token_id = TrainerTools().tokenizer.pad
        device = TrainerTools().parallel.device

        # 左边添加pad，对齐prompt长度
        # [batch, max_prompt_len]
        prompt_ids = left_pad_sequence(prompts, padding_value=pad_token_id)
        prompt_ids = prompt_ids.to(device)

        prompt_len = prompt_ids.shape[1]

        # [batch*group_size, max_prompt_len]
        prompt_ids = prompt_ids.repeat_interleave(group_size, 0)
        # [batch*group_size, max_prompt_len]
        prompt_masks = prompt_ids != pad_token_id

        # [batch*group_size, max_prompt_len+max_gen_len]
        outputs, _ = batch_generate(
            model=model,
            tokens=prompt_ids,
            attention_mask=prompt_masks,
            max_new_tokens=self.grpo_config.gen_max_new_tokens,
            temperature=self.grpo_config.gen_temperature,
            k=self.grpo_config.gen_k,
            p=self.grpo_config.gen_p,
            device=device,
            suppress_tokens=self.grpo_config.gen_suppress_tokens,
            return_logits=False
        )

        # [batch*group_size, max_gen_len]
        completion_ids = outputs[:, prompt_len:]
        # [batch*group_size, max_gen_len]
        completion_masks = (completion_ids != pad_token_id).int()

        return prompt_ids, prompt_masks, completion_ids, completion_masks

    def _generate_rollout_data(self, generate_model, batch_data: List[dict]):
        prompts = [item["prompt"] for item in batch_data]
        answers = [item["answer"] for item in batch_data]
        group_size = self.grpo_config.group_size

        # 使用no_grad替换inference_mode
        # 修复问题：Inference tensors cannot be saved for backward. To work around you can make a clone to get a normal
        with torch.no_grad():
        # with torch.inference_mode():
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_completions(generate_model, prompts, group_size)
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

            old_log_probs, _ = self._compute_log_probs(generate_model, input_ids, attention_mask)

            if self.ref_model:
                ref_log_probs, _ = self._compute_log_probs(self.ref_model, input_ids, attention_mask)
            else:
                ref_log_probs = None

        repeated_prompts = [p for p in prompts for _ in range(group_size)]
        repeated_answers = [a for a in answers for _ in range(group_size)]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'completion_mask': completion_mask,
            'old_log_probs': old_log_probs,
            'ref_log_probs': ref_log_probs,
            'completion_ids': completion_ids,
            'repeated_prompts': repeated_prompts,
            'repeated_answers': repeated_answers,
        }

    def _maximize_grpo_objective(self, rollout_data):
        device = TrainerTools().parallel.device

        input_ids = rollout_data['input_ids']
        attention_mask = rollout_data['attention_mask']
        completion_mask = rollout_data['completion_mask']
        old_log_probs = rollout_data['old_log_probs']
        ref_log_probs = rollout_data['ref_log_probs']
        completion_ids = rollout_data['completion_ids']
        repeated_prompts = rollout_data['repeated_prompts']
        repeated_answers = rollout_data['repeated_answers']

        prompt_len = input_ids.shape[1] - completion_ids.shape[1]

        # [batch*group_size]
        rewards = torch.tensor(
            self.reward_func(repeated_prompts, completion_ids, repeated_answers),
            dtype=torch.float32,
            device=device
        )

        # [batch*group_size, 1]
        advantages = self._compute_group_relative_advantages(rewards)

        # Compute current log probabilities
        log_probs, aux_loss = self._compute_log_probs(self.train_model, input_ids, attention_mask)

        pad_len = prompt_len - 1
        if pad_len > 0:
            padded_completion_mask = F.pad(completion_mask, (pad_len, 0), 'constant', 0)
        else:
            padded_completion_mask = completion_mask

        assert padded_completion_mask.shape == log_probs.shape, \
            f"Shape mismatch! Padded completion mask: {padded_completion_mask.shape}, Log probs: {log_probs.shape}"

        loss = self.criterion(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            completion_mask=padded_completion_mask,
            advantages=advantages
        )

        return loss, aux_loss, rewards

    def train(self):
        global_steps = 0
        skipping_train = False
        aux_loss_coef = self.train_config.loss_config.aux_loss_coef

        for epoch in range(self.train_config.n_epochs):
            if self.ref_model:
                sync_model_params(
                    _from=self.train_model,
                    _to=self.ref_model,
                    mixup_alpha=self.grpo_config.mixup_alpha
                )

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

                    # start generate
                    if TrainerTools().parallel.is_main_process:
                        Logger.std_log(f'start generate for batch {batch}/{batch_count_per_file}')

                    # 生成数据
                    with unwrap_model_for_generation(self.train_model) as generate_model:
                        rollout_data = self._generate_rollout_data(generate_model, batch_data)
                    # end generate

                    torch.cuda.empty_cache()

                    try:
                        if TrainerTools().parallel.is_main_process:
                            Logger.std_log(f'start train for batch {batch}/{batch_count_per_file}')

                        for grpo_step in range(self.grpo_config.grpo_steps):
                            with autocast(TrainerTools().parallel.device_type):
                                loss, aux_loss, rewards = self._maximize_grpo_objective(rollout_data)
                                if aux_loss_coef and aux_loss is not None:
                                    aux_loss = aux_loss_coef * aux_loss
                                else:
                                    aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                            total_loss = loss + aux_loss
                            self._backward_loss(total_loss)
                            self._apply_grad_clipping()
                            self._apply_step()

                            loss_accumulation = total_loss.detach().item()
                            aux_loss_accumulation = aux_loss.detach().item()

                            avg_loss, avg_aux_loss = self._avg_loss(
                                losses=[
                                    loss_accumulation,
                                    aux_loss_accumulation
                                ],
                                gradient_accumulation_steps=1,
                                batches_accumulated=1
                            )

                            self._log(
                                keys={
                                    'epoch': epoch,
                                    'file': f'{file_idx + 1}/{file_count}',
                                    'batch': f'{batch}/{batch_count_per_file}',
                                    'grpo_step': grpo_step
                                },
                                values={
                                    'loss': avg_loss,
                                    'moe_aux_loss': avg_aux_loss,
                                    'rewards': (rewards.sum() / rewards.size(0)).item(),
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
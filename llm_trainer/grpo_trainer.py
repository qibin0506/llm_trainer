from typing import Tuple, List, Union, Callable, Optional
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import torch.nn.functional as F

from .parallel_ds import DsParallel
from .trainer import Trainer
from .train_configs import TrainConfig
from .dataset import GRPORolloutDataset
from .loss import GRPOLoss
from .tools import TrainerTools
from .generate_utils import batch_generate
from .log import log
from .utils import autocast

from .partition_utils import (
    sync_model_params,
    unwrap_model_for_generation
)

from .checkpoint import (
    save_checkpoint,
    save_best_checkpoint,
    save_steps,
)

class GRPOTrainer(Trainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            reward_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], List[float]],
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

        # 默认使用torch提供的pad_sequence
        # 如果pad_sequence不支持padding_side参数，则将改参数置为False，使用反转的方式
        self._use_origin_pad_sequence = True

    def _init_ref_model(self):
        # beta == 0，不需要ref_model
        if self.train_config.grpo_config.loss_beta == 0.0:
            return None

        ref_model = self._new_model(self.train_config)

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
        criterion = GRPOLoss(
            beta=self.train_config.grpo_config.loss_beta,
            clip_eps_low=self.train_config.grpo_config.loss_clip_eps,
            clip_eps_high=self.train_config.grpo_config.loss_clip_eps_high,
            delta=self.train_config.grpo_config.loss_delta,
            importance_sampling_level=self.train_config.grpo_config.loss_importance_sampling_level,
            loss_type=self.train_config.grpo_config.loss_type,
            gen_max_new_tokens=self.train_config.grpo_config.gen_max_new_tokens
        )

        return criterion, None

    def _convert_train_args(self) -> Tuple[dict, dict, dict, bool]:
        parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": lambda x: x})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        return GRPORolloutDataset(file_path), file_path

    def _calc_loss(self, inputs, attention_mask, logits, labels): ...

    def _left_pad_sequence(
            self,
            sequences: Union[torch.Tensor, List[torch.Tensor]],
            padding_value: float,
    ) -> torch.Tensor:
        if self._use_origin_pad_sequence:
            try:
                return pad_sequence(sequences, batch_first=True, padding_value=padding_value, padding_side='left')
            except:
                self._use_origin_pad_sequence = False
                return self._left_pad_sequence(sequences, padding_value)
        else:
            # 反转每个序列的顺序（如 [1,2,3] → [3,2,1]）
            reversed_sequences = [seq.flip(dims=(0,)) for seq in sequences]
            # 使用默认的右侧填充
            padded_reversed = pad_sequence(reversed_sequences, batch_first=True, padding_value=padding_value)
            # 再次反转序列顺序，恢复原始方向（填充在左侧）
            return padded_reversed.flip(dims=(1,))

    def _selective_log_softmax(self, logits, input_ids):
        # Convert raw logits into log probabilities along the vocabulary axis.
        # [batch_size, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

        # Reshape input_ids from (batch_size, seq_len) to (batch_size, seq_len, 1) for gathering.
        # Then, gather the log probability for each token in input_ids.
        selected_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))

        # Remove the extra last dimension to get back to shape (batch_size, seq_len).
        return selected_log_probs.squeeze(-1)

    def _compute_log_probabilities(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep
    ):
        # prompt部分[1, 2, 3]
        # 生成模型生成的内容是[4, 5]，logits_to_keep=2
        # 则下面的输入 [1, 2, 3, 4, 5], 正常情况下输出是[2, 3, 4, 5, 6]
        # logits_to_keep=2，时输出[5, 6]
        # 但是我们想要的[4, 5]部分
        # 所以需要logits_to_keep=2+1，输出[4, 5, 6]

        # [batch_size, total_seq_len, vocab_size]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=logits_to_keep + 1
        )

        # [batch_size, total_seq_len - 1, vocab_size]
        logits = outputs['logits'][:, :-1, :]

        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:, :]

        # Compute and return the log probabilities for the selected tokens.
        return self._selective_log_softmax(logits, input_ids), outputs['aux_loss']

    def _compute_group_relative_advantages(self, rewards):
        group_size = self.train_config.grpo_config.group_size

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
        prompt_ids = self._left_pad_sequence(prompts, padding_value=pad_token_id)
        prompt_ids = prompt_ids.to(device)

        prompt_len = prompt_ids.shape[1]

        # [batch*group_size, max_prompt_len]
        prompt_ids = prompt_ids.repeat_interleave(group_size, 0)
        # [batch*group_size, max_prompt_len]
        prompt_masks = prompt_ids != pad_token_id

        # [batch*group_size, max_prompt_len+max_gen_len]
        outputs: torch.Tensor = batch_generate(
            model=model,
            tokens=prompt_ids,
            pad_token_id=pad_token_id,
            attention_mask=prompt_masks,
            max_position_embeddings=self.train_config.model_config.max_position_embeddings,
            max_new_tokens=self.train_config.grpo_config.gen_max_new_tokens,
            temperature=self.train_config.grpo_config.gen_temperature,
            k=self.train_config.grpo_config.gen_k,
            p=self.train_config.grpo_config.gen_p,
            device=device,
            suppress_tokens=self.train_config.grpo_config.gen_suppress_tokens
        )

        # [batch*group_size, max_gen_len]
        completion_ids = outputs[:, prompt_len:]
        # [batch*group_size, max_gen_len]
        completion_masks = (completion_ids != pad_token_id).int()

        return prompt_ids, prompt_masks, completion_ids, completion_masks

    def _generate_rollout_data(self, generate_model, batch_data: List[dict]):
        prompts = [item["prompt"] for item in batch_data]
        answers = [item["answer"] for item in batch_data]
        group_size = self.train_config.grpo_config.group_size

        # 使用no_grad替换inference_mode
        # 修复问题：Inference tensors cannot be saved for backward. To work around you can make a clone to get a normal
        with torch.no_grad():
        # with torch.inference_mode():
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_completions(generate_model, prompts, group_size)
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.shape[1]

            # Compute old_log_probs from the current model, with gradients disabled.
            old_log_probs, _ = self._compute_log_probabilities(generate_model, input_ids, attention_mask, logits_to_keep)

            if self.ref_model:
                # Compute ref_log_probs from the reference model, which remains static.
                ref_log_probs, _ = self._compute_log_probabilities(self.ref_model, input_ids, attention_mask, logits_to_keep)
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
            'logits_to_keep': logits_to_keep
        }

    def _maximize_grpo_objective(self, rollout_data):
        device = TrainerTools().parallel.device

        input_ids = rollout_data['input_ids']
        attention_mask = rollout_data['attention_mask']
        completion_mask = rollout_data['completion_mask']
        old_log_probs = rollout_data['old_log_probs']
        ref_log_probs = rollout_data['ref_log_probs']
        logits_to_keep = rollout_data['logits_to_keep']
        completion_ids = rollout_data['completion_ids']
        repeated_prompts = rollout_data['repeated_prompts']
        repeated_answers = rollout_data['repeated_answers']

        # [batch*group_size]
        rewards = torch.tensor(
            self.reward_func(repeated_prompts, completion_ids, repeated_answers),
            dtype=torch.float32,
            device=device
        )

        # [batch*group_size, 1]
        advantages = self._compute_group_relative_advantages(rewards)

        # Compute current log probabilities
        log_probs, aux_loss = self._compute_log_probabilities(self.train_model, input_ids, attention_mask, logits_to_keep)

        loss = self.criterion(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            completion_mask=completion_mask,
            advantages=advantages
        )

        return loss, aux_loss

    def train(self):
        global_steps = 0
        skipping_train = False

        current_loss: float = 0.0
        last_best_checkpoint_loss: Optional[float] = None

        aux_loss_coef = self.train_config.loss_config.aux_loss_coef

        for epoch in range(self.train_config.n_epochs):
            if self.ref_model:
                sync_model_params(
                    _from=self.train_model,
                    _to=self.ref_model,
                    mixup_alpha=self.train_config.grpo_config.mixup_alpha
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
                        log(f'start generate for batch {batch}/{batch_count_per_file}')

                    # 生成数据
                    with unwrap_model_for_generation(self.train_model) as generate_model:
                        rollout_data = self._generate_rollout_data(generate_model, batch_data)

                    torch.cuda.empty_cache()
                    # end generate

                    try:
                        if TrainerTools().parallel.is_main_process:
                            log(f'start train for batch {batch}/{batch_count_per_file}')

                        for grpo_step in range(self.train_config.grpo_config.grpo_steps):
                            with autocast(TrainerTools().parallel.device_type):
                                loss, aux_loss = self._maximize_grpo_objective(rollout_data)
                                if aux_loss_coef and aux_loss:
                                    loss += aux_loss_coef * aux_loss

                            self._backward_loss(loss)

                            if TrainerTools().parallel.parallel_train:
                                dist.all_reduce(loss, dist.ReduceOp.AVG)

                            current_loss = loss.detach().item()

                            # ds模式已经集成gradient_clipping
                            if not isinstance(TrainerTools().parallel, DsParallel) and self.lr_scheduler.can_clip_grad():
                                # clip grad
                                self.scalar.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self._get_trainable_params(self.train_model), 1.0)

                            self._step()

                            self._log_loss(
                                epoch_tag=f'epoch: {epoch}',
                                file_tag=f'file: {file_idx + 1}/{file_count}',
                                batch_tag=f'batch: {batch}/{batch_count_per_file}, grpo_step={grpo_step}',
                                loss=current_loss
                            )
                    except Exception as e:
                        self._on_exception(e, epoch, batch)
                    finally:
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
from typing import Tuple, List, Optional
import gc
import math
from itertools import islice

import torch
from torch.utils.data import Dataset
import torch.distributed as dist

from .base_trainer import BaseTrainer
from .dataset import DPODataset
from .loss import DPOLoss, log1mexp
from .tools import TrainerTools
from .log import Logger
from .train_configs import TrainConfig, GenerationService
from .utils import autocast, get_dpo_collate_fn, log_softmax, disable_dropout_in_model, empty_cache
from .checkpoint import save_checkpoint, save_steps, restore_rng_state_dict
from .partition_utils import zero3_trace_guard, add_fused_log_prob_head


class DPOTrainer(BaseTrainer):
    """
    DPOTrainer

    Args:
        train_config:
            - 全局训练配置，必须包含 dpo_config。

        eval_prompts:
            - 评估阶段生成测试的提示词列表。
            - [num_eval_prompts] 长度的字符串列表。

        generation_service:
            - 外部自定义生成服务接口。
    """
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            generation_service: Optional[GenerationService] = None,
    ):
        self.dpo_config = train_config.dpo_config
        self.chunked_log_probs_size = self.dpo_config.chunked_log_probs_size
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            generation_service=generation_service,
            gradient_accumulation_steps=self.dpo_config.gradient_accumulation_steps
        )
        self.ref_model = self._init_ref_model()
        self.criterion = self._init_loss()

        if self.dpo_config.loss_type == 'orpo' and (self.dpo_config.nll_loss_coef is None or self.dpo_config.nll_loss_coef == 0.0):
            self.dpo_config.nll_loss_coef = 1.0
            if TrainerTools().parallel.is_main_process:
                Logger.std_log("WARN: ORPO highly recommends using an NLL (SFT) loss. Please set `nll_loss_coef` (e.g., 1.0) in DPOConfig!")

    def _init_ref_model(self):
        if self.dpo_config.loss_type in ['orpo', 'simpo']:
            return None

        if not self.dpo_config.ref_model_weights_path:
            raise ValueError(
                f"DPO (loss_type='{self.dpo_config.loss_type}') requires a reference model to compute preference loss, "
                "but `ref_model_weights_path` is not configured! "
                "Please configure `ref_model_weights_path` in DPOConfig to prevent using randomly initialized weights in distributed training."
            )

        parallel_kwargs = self._init_ref_model_args(self.train_config.model_config)
        with self._new_model_context(parallel_kwargs):
            ref_model = self._new_model(self.train_config)

        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

        self._check_add_fused_head(ref_model)
        self._load_external_weights(ref_model, self.dpo_config.ref_model_weights_path)

        ref_model, _ = TrainerTools().parallel.process(
            model=ref_model,
            optimizer=None,
            kwargs=parallel_kwargs,
            save_instance=False
        )

        return ref_model

    def _check_add_fused_head(self, model):
        if self.chunked_log_probs_size is not None:
            add_fused_log_prob_head(model, chunk_size=self.chunked_log_probs_size)

    def _new_model(self, train_config: TrainConfig):
        model = super()._new_model(train_config)
        disable_dropout_in_model(model)
        return model

    def _init_loss(self):
        return DPOLoss(
            beta=self.dpo_config.loss_beta,
            label_smoothing=self.dpo_config.loss_label_smoothing,
            ipo=self.dpo_config.loss_ipo,
            loss_type=self.dpo_config.loss_type,
            simpo_gamma=self.dpo_config.simpo_gamma
        )

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        dpo_collate_fn = get_dpo_collate_fn(self.dpo_config.mask_prompt)
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()

        if parallel_kwargs:
            # 因为chosen和inject会concat到一块进行forward，所以实际batch_size要*2
            real_micro_batch_size = self.train_config.batch_size * 2
            parallel_kwargs['train_micro_batch_size_per_gpu'] = real_micro_batch_size

        data_loader_kwargs.update({"collate_fn": dpo_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        block_size = self.train_config.dataset_block_size
        return DPODataset(file_path, block_size), file_path

    def _logprobs(self, logits_or_logprobs, labels, is_logprobs: bool = False):
        """
        Calculate the average log probabilities for a batch of sequences.

        Args:
            logits_or_logprobs (torch.Tensor): Logits from model (B, T, V) or log_probs (B, T-1).
            labels (torch.Tensor): Ground truth labels with shape (B, T).
            is_logprobs (bool): Whether logits_or_logprobs is already per-token log_probs.

        Returns:
            torch.Tensor: Average log probabilities for each sequence in the batch.
        """
        loss_masks = (labels != -100)
        loss_masks = loss_masks[:, 1:]

        if is_logprobs:
            per_token_logps = logits_or_logprobs
        else:
            logits = logits_or_logprobs[:, :-1, :]
            shift_labels = labels[:, 1:].clone()
            per_token_logps = log_softmax(logits, shift_labels)

        # Apply the mask to set log-probs of padding/prompt tokens to 0
        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1).clamp(min=1.0)
        mask_sums = loss_masks.sum(-1)

        return logprobs_sums, logprobs_means, mask_sums

    def train(self):
        loss_accumulation = 0.0
        dpo_loss_accumulation = 0.0
        aux_loss_accumulation = 0.0
        nll_loss_accumulation = 0.0
        ce_loss_accumulation = 0.0
        chosen_reward_accumulation = 0.0
        rejected_reward_accumulation = 0.0
        reward_margin_accumulation = 0.0
        reward_accuracy_accumulation = 0.0
        global_steps_since_last_save = 0
        global_steps_since_last_eval = 0
        batches_accumulated = 0

        nll_loss_coef = self.dpo_config.nll_loss_coef
        beta = self.dpo_config.loss_beta

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
                TrainerTools().parallel.on_epoch_start(epoch, file_idx)
                self._on_file_start(epoch, file_path)

                skip_batches = 0
                if epoch == self.resume_epoch and file_idx == self.resume_file_idx:
                    skip_batches = self.resume_batch_idx
                    if skip_batches > 0 and TrainerTools().parallel.is_main_process:
                        Logger.std_log(f"Fast forwarding {skip_batches} batches in {file_path}...")

                if skip_batches > 0 and self.resume_dataloader_rng_state is not None:
                    torch.set_rng_state(self.resume_dataloader_rng_state.cpu().to(torch.uint8))
                    self.resume_dataloader_rng_state = None

                dataloader_rng_state = torch.get_rng_state()
                data_iterator = iter(train_data_loader)
                if skip_batches > 0:
                    data_iterator = islice(data_iterator, skip_batches, None)

                for batch, batch_data in enumerate(data_iterator):
                    if self.resume_rng_state is not None:
                        restore_rng_state_dict(self.resume_rng_state)
                        self.resume_rng_state = None

                    batch = skip_batches + batch

                    try:
                        chosen_inputs: torch.Tensor = batch_data['chosen_inputs'].to(TrainerTools().parallel.device)
                        chosen_labels: torch.Tensor = batch_data['chosen_labels'].to(TrainerTools().parallel.device)

                        rejected_inputs: torch.Tensor = batch_data['rejected_inputs'].to(TrainerTools().parallel.device)
                        rejected_labels: torch.Tensor = batch_data['rejected_labels'].to(TrainerTools().parallel.device)

                        chosen_attention_masks: torch.Tensor = self._calc_attention_mask(chosen_inputs)
                        rejected_attention_masks: torch.Tensor = self._calc_attention_mask(rejected_inputs)

                        # 在batch维度concat
                        # [chosen, chosen, reject, reject]
                        concat_inputs = torch.concat([chosen_inputs, rejected_inputs], dim=0)
                        concat_labels = torch.concat([chosen_labels, rejected_labels], dim=0)
                        concat_attention_masks = torch.concat([chosen_attention_masks, rejected_attention_masks], dim=0)

                        use_fused = (self.chunked_log_probs_size is not None)
                        with autocast(TrainerTools().parallel.device_type):
                            if self.ref_model is not None:
                                with torch.no_grad():
                                    with zero3_trace_guard(self.ref_model):
                                        ref_outputs = self.ref_model(
                                            concat_inputs,
                                            attention_mask=concat_attention_masks,
                                            labels=concat_labels if use_fused else None,
                                            fused_log_probs=use_fused,
                                            return_logits=not use_fused
                                        )
                                        ref_data = ref_outputs['log_probs'] if use_fused else ref_outputs['logits']
                                        ref_logprobs_sums, ref_logprobs_means, _ = self._logprobs(ref_data, concat_labels, is_logprobs=use_fused)
                                        del ref_outputs

                                    ref_chosen_logps = ref_logprobs_sums[:chosen_inputs.shape[0]]
                                    ref_rejected_logps = ref_logprobs_sums[chosen_inputs.shape[0]:]
                                    ref_chosen_means = ref_logprobs_means[:chosen_inputs.shape[0]]
                                    ref_rejected_means = ref_logprobs_means[chosen_inputs.shape[0]:]
                            else:
                                ref_chosen_logps = None
                                ref_rejected_logps = None
                                ref_chosen_means = None
                                ref_rejected_means = None

                            policy_outputs = self.train_model(
                                concat_inputs,
                                attention_mask=concat_attention_masks,
                                labels=concat_labels if use_fused else None,
                                fused_log_probs=use_fused,
                                return_logits=not use_fused
                            )
                            policy_data = policy_outputs['log_probs'] if use_fused else policy_outputs['logits']
                            policy_logprobs_sums, policy_logprobs_means, policy_mask_sums = self._logprobs(policy_data, concat_labels, is_logprobs=use_fused)

                            aux_loss = policy_outputs['aux_loss']
                            del policy_outputs

                            policy_chosen_logps = policy_logprobs_sums[:chosen_inputs.shape[0]]
                            policy_rejected_logps = policy_logprobs_sums[chosen_inputs.shape[0]:]

                            policy_chosen_means = policy_logprobs_means[:chosen_inputs.shape[0]]
                            policy_rejected_means = policy_logprobs_means[chosen_inputs.shape[0]:]
                            chosen_mask_sums = policy_mask_sums[:chosen_inputs.shape[0]]
                            rejected_mask_sums = policy_mask_sums[chosen_inputs.shape[0]:]

                            # 采用标准 Token-level 交叉熵损失 (Token-level NLL)，避免“均值的均值”导致长短样本 token 梯度失衡
                            local_chosen_tokens = chosen_mask_sums.sum().float()
                            if TrainerTools().parallel.parallel_train and torch.distributed.is_initialized():
                                global_chosen_tokens = local_chosen_tokens.clone()
                                dist.all_reduce(global_chosen_tokens, op=torch.distributed.ReduceOp.SUM)
                                world_size = TrainerTools().parallel.world_size
                                nll_loss = -(policy_chosen_logps.sum() * world_size / global_chosen_tokens.clamp(min=1.0))
                            else:
                                nll_loss = -(policy_chosen_logps.sum() / local_chosen_tokens.clamp(min=1.0))

                            ce_loss = nll_loss.detach()

                            # calc loss
                            loss = self.criterion(
                                policy_chosen_logps=policy_chosen_logps,
                                policy_reject_logps=policy_rejected_logps,
                                ref_chosen_logps=ref_chosen_logps,
                                ref_reject_logps=ref_rejected_logps,
                                policy_chosen_means=policy_chosen_means,
                                policy_reject_means=policy_rejected_means,
                                chosen_lengths=chosen_mask_sums,
                                reject_lengths=rejected_mask_sums,
                                ref_chosen_means=ref_chosen_means,
                                ref_reject_means=ref_rejected_means,
                            )

                            if aux_loss is not None:
                                aux_loss = aux_loss.to(loss.dtype)
                            else:
                                aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                            if nll_loss_coef is not None and nll_loss_coef != 0.0:
                                nll_loss = nll_loss_coef * nll_loss
                            else:
                                nll_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                            with torch.no_grad():
                                if self.dpo_config.loss_type == 'orpo':
                                    p_c = torch.clamp(policy_chosen_means.float(), max=-1e-7)
                                    p_r = torch.clamp(policy_rejected_means.float(), max=-1e-7)
                                    chosen_rewards = torch.where(chosen_mask_sums > 0, p_c - log1mexp(p_c), torch.zeros_like(p_c))
                                    rejected_rewards = torch.where(rejected_mask_sums > 0, p_r - log1mexp(p_r), torch.zeros_like(p_r))
                                elif self.dpo_config.loss_type == 'simpo':
                                    chosen_rewards = beta * policy_chosen_means
                                    rejected_rewards = beta * policy_rejected_means
                                else:
                                    # DPO Margin = Beta * (Policy_logps - Ref_logps)
                                    chosen_rewards = beta * (policy_chosen_logps.detach() - ref_chosen_logps)
                                    rejected_rewards = beta * (policy_rejected_logps.detach() - ref_rejected_logps)

                                reward_margin = chosen_rewards - rejected_rewards
                                reward_accuracy = (chosen_rewards > rejected_rewards).float()

                        total_loss_unscaled = loss + aux_loss + nll_loss

                        window_start = (batch // self.gradient_accumulation_steps) * self.gradient_accumulation_steps
                        actual_accum_steps = min(self.gradient_accumulation_steps, batch_count_per_file - window_start)

                        # 每个文件的最后一个 batch 作为累积边界强制提交梯度，确保每个文件起点 batches_accumulated 恒为 0，
                        is_last_step = (batch == batch_count_per_file - 1)

                        need_update_step = self._need_update_step(batches_accumulated, is_last_step)
                        if self.is_ds:
                            if actual_accum_steps < self.gradient_accumulation_steps:
                                scale_factor = self.gradient_accumulation_steps / actual_accum_steps
                                self._backward_loss(total_loss_unscaled * scale_factor, actual_accum_steps)
                            else:
                                self._backward_loss(total_loss_unscaled, self.gradient_accumulation_steps)
                        else:
                            self._backward_loss(total_loss_unscaled, actual_accum_steps)

                        loss_accumulation += total_loss_unscaled.detach().item()
                        dpo_loss_accumulation += loss.detach().item()
                        aux_loss_accumulation += aux_loss.detach().item()
                        nll_loss_accumulation += nll_loss.detach().item()
                        ce_loss_accumulation += ce_loss.detach().item()

                        chosen_reward_accumulation += chosen_rewards.mean().item()
                        rejected_reward_accumulation += rejected_rewards.mean().item()
                        reward_margin_accumulation += reward_margin.mean().item()
                        reward_accuracy_accumulation += reward_accuracy.mean().item()

                        batches_accumulated += 1

                        if need_update_step:
                            self._update_step(is_last_step=is_last_step)
                            global_steps_since_last_save += 1
                            global_steps_since_last_eval += 1

                            avg_total_loss, avg_dpo_loss, \
                            avg_aux_loss, avg_nll_loss, avg_ce_loss, \
                            avg_chosen_reward, avg_rejected_reward, \
                            avg_reward_margin, avg_reward_accuracy = self._avg_loss(
                                losses=[
                                    loss_accumulation,
                                    dpo_loss_accumulation,
                                    aux_loss_accumulation,
                                    nll_loss_accumulation,
                                    ce_loss_accumulation,
                                    chosen_reward_accumulation,
                                    rejected_reward_accumulation,
                                    reward_margin_accumulation,
                                    reward_accuracy_accumulation
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
                                    'batch': f'{batch + 1}/{batch_count_per_file}',
                                },
                                values={
                                    'loss/total': avg_total_loss,
                                    f'loss/{self.dpo_config.loss_type}': avg_dpo_loss,
                                    'loss/moe_aux': avg_aux_loss,
                                    'loss/nll': avg_nll_loss,
                                    'metrics/ppl': round(perplexity, 4) if avg_ce_loss > 0 else float('inf'),
                                    'reward/chosen': avg_chosen_reward,
                                    'reward/rejected': avg_rejected_reward,
                                    'reward/margin': avg_reward_margin,
                                    'reward/accuracy': avg_reward_accuracy
                                }
                            )

                            loss_accumulation = 0.0
                            dpo_loss_accumulation = 0.0
                            aux_loss_accumulation = 0.0
                            nll_loss_accumulation = 0.0
                            ce_loss_accumulation = 0.0
                            chosen_reward_accumulation = 0.0
                            rejected_reward_accumulation = 0.0
                            reward_margin_accumulation = 0.0
                            reward_accuracy_accumulation = 0.0
                            batches_accumulated = 0

                            if 0 < self.train_config.save_interval <= global_steps_since_last_save:
                                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                                save_steps(
                                    epoch=epoch,
                                    file_idx=file_idx,
                                    batch_idx=batch + 1,
                                    lr_scheduler=self.lr_scheduler,
                                    dataloader_rng_state=dataloader_rng_state
                                )
                                global_steps_since_last_save = 0

                            if 0 < self.train_config.eval_interval <= global_steps_since_last_eval:
                                self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')
                                global_steps_since_last_eval = 0
                    except Exception as e:
                        self._on_exception(e, epoch, batch)

                if self.resume_rng_state is not None:
                    restore_rng_state_dict(self.resume_rng_state)
                    self.resume_rng_state = None

                try:
                    del train_data_loader
                    del dataset
                    del data_iterator
                    del dataloader_rng_state
                    del batch_data
                    del concat_inputs
                    del concat_labels
                    del concat_attention_masks
                    del loss
                    del aux_loss
                    del nll_loss
                    del ce_loss
                    del total_loss_unscaled
                    del chosen_inputs
                    del chosen_labels
                    del rejected_inputs
                    del rejected_labels
                    del chosen_attention_masks
                    del rejected_attention_masks
                    del policy_chosen_logps
                    del policy_rejected_logps
                    del policy_chosen_means
                    del policy_rejected_means
                    del ref_chosen_logps
                    del ref_rejected_logps
                    del chosen_rewards
                    del rejected_rewards
                    del reward_margin
                    del reward_accuracy
                except (NameError, UnboundLocalError): ...

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

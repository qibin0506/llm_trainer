from typing import Tuple, List
import gc
import math
import torch
from torch.utils.data import Dataset
from itertools import islice

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig
from .dataset import DPODataset
from .loss import DPOLoss
from .tools import TrainerTools
from .utils import (
    autocast,
    get_dpo_collate_fn,
    log_softmax,
    disable_dropout_in_model,
    empty_cache
)

from .checkpoint import (
    save_checkpoint,
    save_steps,
)
from .log import Logger


class DPOTrainer(BaseTrainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str]
    ):
        self.dpo_config = train_config.dpo_config
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            gradient_accumulation_steps=self.dpo_config.gradient_accumulation_steps
        )
        self.ref_model = self._init_ref_model()

    def _init_ref_model(self):
        ref_model = self._new_model(self.train_config)

        if self.dpo_config.ref_model_checkpoint:
            ref_model.load_state_dict(self.dpo_config.ref_model_checkpoint)
            self.dpo_config.ref_model_checkpoint = {}

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
        criterion = DPOLoss(
            beta=self.dpo_config.loss_beta,
            label_smoothing=self.dpo_config.loss_label_smoothing,
            ipo=self.dpo_config.loss_ipo
        )

        return criterion, None

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

    def _calc_loss(self, inputs, attention_mask, logits, labels): ...

    def _logprobs(self, logits, labels):
        """
        Calculate the average log probabilities for a batch of sequences.

        Args:
            logits (torch.Tensor): Logits from the model with shape (B, T, V)
            labels (torch.Tensor): Ground truth labels with shape (B, T).

        Returns:
            torch.Tensor: Average log probabilities for each sequence in the batch.
                          Shape is (B,) representing the mean log probability for each sequence.
        """
        loss_masks = (labels != -100)

        logits = logits[:, :-1, :]
        labels = labels[:, 1:].clone()
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        # Gather the log probabilities for the actual labels
        per_token_logps = log_softmax(logits, labels)

        # Apply the mask to set log-probs of padding tokens to 0
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

        batches_accumulated = 0

        aux_loss_coef = self.train_config.loss_config.aux_loss_coef
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

                last_ckpt_batch = 0
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
                    last_ckpt_batch = skip_batches

                for batch, batch_data in enumerate(data_iterator):
                    batch = skip_batches + batch

                    try:
                        chosen_inputs: torch.Tensor = batch_data['chosen_inputs'].to(TrainerTools().parallel.device)
                        chosen_labels: torch.Tensor = batch_data['chosen_labels'].to(TrainerTools().parallel.device)

                        rejected_inputs: torch.Tensor = batch_data['rejected_inputs'].to(TrainerTools().parallel.device)
                        rejected_labels: torch.Tensor = batch_data['rejected_labels'].to(TrainerTools().parallel.device)

                        chosen_attention_masks: torch.Tensor = chosen_inputs != TrainerTools().tokenizer.pad
                        rejected_attention_masks: torch.Tensor = rejected_inputs != TrainerTools().tokenizer.pad

                        # 在batch维度concat
                        # [chosen, chosen, reject, reject]
                        concat_inputs = torch.concat([chosen_inputs, rejected_inputs], dim=0)
                        concat_labels = torch.concat([chosen_labels, rejected_labels], dim=0)
                        concat_attention_masks = torch.concat([chosen_attention_masks, rejected_attention_masks], dim=0)

                        with autocast(TrainerTools().parallel.device_type):
                            with torch.no_grad():
                                ref_outputs = self.ref_model(concat_inputs, attention_mask=concat_attention_masks)
                                ref_logprobs_sums, _, _ = self._logprobs(ref_outputs['logits'], concat_labels)
                                del ref_outputs

                            policy_outputs = self.train_model(concat_inputs, attention_mask=concat_attention_masks)
                            policy_logprobs_sums, policy_logprobs_means, policy_mask_sums = self._logprobs(policy_outputs['logits'], concat_labels)

                            raw_aux_loss = policy_outputs.get('aux_loss', None)
                            del policy_outputs

                            policy_chosen_logps = policy_logprobs_sums[:chosen_inputs.shape[0]]
                            policy_rejected_logps = policy_logprobs_sums[chosen_inputs.shape[0]:]

                            ref_chosen_logps = ref_logprobs_sums[:chosen_inputs.shape[0]]
                            ref_rejected_logps = ref_logprobs_sums[chosen_inputs.shape[0]:]

                            nll_loss = -policy_logprobs_means[:chosen_inputs.shape[0]].mean()

                            chosen_logprobs_sums = policy_logprobs_sums[:chosen_inputs.shape[0]]
                            chosen_mask_sums = policy_mask_sums[:chosen_inputs.shape[0]]
                            ce_loss = -(chosen_logprobs_sums.sum() / chosen_mask_sums.sum().clamp(min=1.0))

                            # calc loss
                            loss = self.criterion(
                                policy_chosen_logps,
                                policy_rejected_logps,
                                ref_chosen_logps,
                                ref_rejected_logps
                            )

                            if aux_loss_coef and raw_aux_loss is not None:
                                aux_loss = aux_loss_coef * raw_aux_loss
                            else:
                                aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                            if nll_loss_coef and nll_loss:
                                nll_loss = nll_loss_coef * nll_loss
                            else:
                                nll_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                            with torch.no_grad():
                                chosen_rewards = beta * (policy_chosen_logps.detach() - ref_chosen_logps)
                                rejected_rewards = beta * (policy_rejected_logps.detach() - ref_rejected_logps)
                                reward_margin = chosen_rewards - rejected_rewards
                                reward_accuracy = (chosen_rewards > rejected_rewards).float()

                        total_loss_unscaled = loss + aux_loss + nll_loss

                        is_last_step = (
                            epoch == self.train_config.n_epochs - 1 and
                            file_idx == file_count - 1 and
                            batch == batch_count_per_file - 1
                        )
                        need_update_step = self._need_update_step(batches_accumulated, is_last_step)
                        self._backward_loss(total_loss_unscaled, self.gradient_accumulation_steps)

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
                            self._update_step()

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
                                    'loss/dpo': avg_dpo_loss,
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

                            if (batch - last_ckpt_batch) >= self.train_config.eval_config.eval_batch_interval:
                                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                                save_steps(
                                    epoch=epoch,
                                    file_idx=file_idx,
                                    batch_idx=batch + 1,
                                    lr_scheduler=self.lr_scheduler
                                )

                                last_ckpt_batch = batch
                                self._on_batch_end(tag=f'epoch:{epoch}/batch:{batch}')
                    except Exception as e:
                        self._on_exception(e, epoch, batch)

                try:
                    del train_data_loader
                    del dataset
                    del data_iterator
                    del batch_data
                    del concat_inputs
                    del concat_labels
                    del concat_attention_masks
                    del loss
                    del aux_loss
                    del nll_loss
                    del total_loss_unscaled
                    del chosen_inputs
                    del chosen_labels
                    del rejected_inputs
                    del rejected_labels
                    del chosen_attention_masks
                    del rejected_attention_masks
                    del policy_chosen_logps
                    del policy_rejected_logps
                    del ref_chosen_logps
                    del ref_rejected_logps
                    del chosen_rewards
                    del rejected_rewards
                    del reward_margin
                    del reward_accuracy
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

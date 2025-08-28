from typing import Tuple, List, Optional
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import torch.nn.functional as F

from .parallel_ds import DsParallel
from .trainer import Trainer
from .train_configs import TrainConfig
from .dataset import DPODataset
from .loss import DPOLoss
from .tools import TrainerTools
from .utils import (
    autocast,
    get_dpo_collate_fn,
    fill_loss_mask
)
from .partition_utils import sync_model_params

from .checkpoint import (
    save_checkpoint,
    save_best_checkpoint,
    save_steps,
)


class DPOTrainer(Trainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            eval_prompts: List[str],
            eval_image_tags: Optional[List[str]] = None
    ):
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts,
            eval_image_tags=eval_image_tags
        )
        self.packed_sequences = False
        self.ref_model = self._init_ref_model()

    def _init_ref_model(self):
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

        sync_model_params(
            _from=self.train_model,
            _to=ref_model
        )

        return ref_model

    def _init_loss(self):
        criterion = DPOLoss(
            beta=self.train_config.dpo_config.loss_beta,
            label_smoothing=self.train_config.dpo_config.loss_label_smoothing,
            ipo=self.train_config.dpo_config.loss_ipo
        )

        return criterion, None

    def _convert_train_args(self) -> Tuple[dict, dict, dict, bool]:
        dpo_collate_fn = get_dpo_collate_fn(self.train_config.mask_prompt)
        parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim = super()._convert_train_args()
        data_loader_kwargs.update({"collate_fn": dpo_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs, use_ds_optim

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        max_position_embeddings = self.train_config.model_config.max_position_embeddings
        return DPODataset(file_path, max_position_embeddings), file_path

    def _calc_loss(self, inputs, attention_mask, logits, labels): ...

    def _log_probs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dtype in [torch.float32, torch.float64]:
            logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.stack(
                [torch.logsumexp(l, dim=-1) for l in logits]  # loop to reduce peak mem consumption
            )
            log_probs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
        else:
            log_probs_labels = []
            for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
                row_log_probs = F.log_softmax(row_logits, dim=-1)
                row_log_probs_labels = row_log_probs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                log_probs_labels.append(row_log_probs_labels)
            log_probs_labels = torch.stack(log_probs_labels)

        return log_probs_labels


    def _logprobs(self, logits, labels, attention_mask):
        """
        Calculate the average log probabilities for a batch of sequences.

        Args:
            logits (torch.Tensor): Logits from the model with shape (B, T, V)
            labels (torch.Tensor): Ground truth labels with shape (B, T).
            attention_mask (torch.Tensor): Mask tensor with shape (B, T) indicating
                which tokens are not padding (1 for valid tokens, 0 for padding).

        Returns:
            torch.Tensor: Average log probabilities for each sequence in the batch.
                          Shape is (B,) representing the mean log probability for each sequence.
        """
        loss_masks = attention_mask.clone().bool()
        loss_masks = fill_loss_mask(loss_masks, labels)

        logits = logits[:, :-1, :]
        labels = labels[:, 1:].clone()
        loss_masks = loss_masks[:, 1:]

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == -100] = 0

        # Gather the log probabilities for the actual labels
        per_token_logps = self._log_probs_from_logits(logits, labels)

        # Apply the mask to set log-probs of padding tokens to 0
        logprobs_sums = (per_token_logps * loss_masks).sum(-1)
        logprobs_means = (per_token_logps * loss_masks).sum(-1) / loss_masks.sum(-1)

        return logprobs_sums, logprobs_means

    def train(self):
        # 梯度累积步数
        gradient_accumulation_steps = self.train_config.gradient_accumulation_steps
        global_steps = 0
        skipping_train = False

        loss_accumulation = 0.0
        batches_accumulated = 0
        current_loss: float = 0.0
        last_best_checkpoint_loss: Optional[float] = None

        aux_loss_coef = self.train_config.loss_config.aux_loss_coef
        nll_loss_coef = self.train_config.dpo_config.nll_loss_coef

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

                        if TrainerTools().parallel.parallel_train:
                            self.train_model.require_backward_grad_sync = need_update_grad

                        with autocast(TrainerTools().parallel.device_type):
                            policy_outputs = self.train_model(concat_inputs, attention_mask=concat_attention_masks)
                            policy_logprobs_sums, policy_logprobs_means = self._logprobs(policy_outputs['logits'], concat_labels, concat_attention_masks)
                            aux_loss = policy_outputs.get('aux_loss')

                            with torch.no_grad():
                                ref_outputs = self.ref_model(concat_inputs, attention_mask=concat_attention_masks)
                                ref_logprobs_sums, _ = self._logprobs(ref_outputs['logits'], concat_labels, concat_attention_masks)

                            policy_chosen_logps = policy_logprobs_sums[:chosen_inputs.shape[0]]
                            policy_rejected_logps = policy_logprobs_sums[chosen_inputs.shape[0]:]

                            ref_chosen_logps = ref_logprobs_sums[:chosen_inputs.shape[0]]
                            ref_rejected_logps = ref_logprobs_sums[chosen_inputs.shape[0]:]

                            nll_loss = -policy_logprobs_means[:chosen_inputs.shape[0]].mean()

                            # calc loss
                            loss = self.criterion(
                                policy_chosen_logps,
                                policy_rejected_logps,
                                ref_chosen_logps,
                                ref_rejected_logps
                            )

                            if aux_loss_coef and aux_loss:
                                loss += aux_loss_coef * aux_loss

                            if nll_loss_coef and nll_loss:
                                loss += nll_loss_coef * nll_loss

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


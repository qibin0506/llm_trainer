from typing import Tuple, List, Optional
import gc
import torch
from torch.utils.data import Dataset

from .base_trainer import BaseTrainer
from .train_configs import TrainConfig
from .dataset import DPODataset
from .loss import DPOLoss
from .tools import TrainerTools
from .utils import (
    autocast,
    get_dpo_collate_fn,
    log_softmax,
    disable_dropout_in_model
)

from .checkpoint import (
    save_checkpoint,
    save_steps,
)


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
        data_loader_kwargs.update({"collate_fn": dpo_collate_fn})

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_idx) -> Tuple[Dataset, str]:
        file_path = self.train_config.file_dataset[file_idx]
        max_seq_len = self.train_config.max_seq_len
        return DPODataset(file_path, max_seq_len), file_path

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

        return logprobs_sums, logprobs_means

    def train(self):
        # 梯度累积步数
        gradient_accumulation_steps = max(1, self.gradient_accumulation_steps)
        global_steps = 0
        skipping_train = False

        loss_accumulation = 0.0
        aux_loss_accumulation = 0.0
        nll_loss_accumulation = 0.0
        batches_accumulated = 0

        aux_loss_coef = self.train_config.loss_config.aux_loss_coef
        nll_loss_coef = self.dpo_config.nll_loss_coef

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
                            policy_logprobs_sums, policy_logprobs_means = self._logprobs(policy_outputs['logits'], concat_labels)

                            with torch.no_grad():
                                ref_outputs = self.ref_model(concat_inputs, attention_mask=concat_attention_masks)
                                ref_logprobs_sums, _ = self._logprobs(ref_outputs['logits'], concat_labels)

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

                            if aux_loss_coef and policy_outputs.get('aux_loss'):
                                aux_loss = aux_loss_coef * policy_outputs.get('aux_loss')
                            else:
                                aux_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                            if nll_loss_coef and nll_loss:
                                nll_loss = nll_loss_coef * nll_loss
                            else:
                                nll_loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps
                            aux_loss = aux_loss / gradient_accumulation_steps
                            nll_loss = nll_loss / gradient_accumulation_steps

                        total_loss = loss + aux_loss + nll_loss
                        self._backward_loss(total_loss)

                        loss_accumulation += total_loss.detach().item()
                        aux_loss_accumulation += aux_loss.detach().item()
                        nll_loss_accumulation += nll_loss.detach().item()

                        batches_accumulated += 1

                        if need_update_grad:
                            self._apply_grad_clipping()
                            self._apply_step()

                            avg_loss, avg_aux_loss, avg_nll_loss = self._avg_loss(
                                losses=[
                                    loss_accumulation,
                                    aux_loss_accumulation,
                                    nll_loss_accumulation,
                                ],
                                gradient_accumulation_steps=gradient_accumulation_steps,
                                batches_accumulated=batches_accumulated
                            )

                            self._log(
                                keys={
                                    'epoch': epoch,
                                    'file': f'{file_idx + 1}/{file_count}',
                                    'batch': f'{batch}/{batch_count_per_file}',
                                },
                                values={
                                    'loss': avg_loss,
                                    'moe_aux_loss': avg_aux_loss,
                                    'nll_loss': avg_nll_loss
                                }
                            )

                            # reset to default
                            loss_accumulation = 0.0
                            aux_loss_accumulation = 0.0
                            nll_loss_accumulation = 0.0
                            batches_accumulated = 0
                    except Exception as e:
                        self._on_exception(e, epoch, batch)
                    finally:
                        if need_update_grad:
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


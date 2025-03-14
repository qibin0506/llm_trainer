import time
from typing import Tuple, List, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from llama import LlamaModel

from .parallel_ds import DsParallel
from .trainer import Trainer
from .train_configs import TrainConfig
from .dataset import GRPORolloutDataset, GRPODataset
from .loss import GRPOLoss
from .tools import TrainerTools
from .generate_utils import batch_generate
from .utils import join_batch
from .log import log

from .checkpoint import (
    save_checkpoint,
    load_checkpoint_for_eval,
    save_steps,
)

class GRPOTrainer(Trainer):
    def __init__(
            self,
            *,
            train_config: TrainConfig,
            reward_funcs: List[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], float]],
            eval_prompts: List[str]
    ):
        super().__init__(
            train_config=train_config,
            eval_prompts=eval_prompts
        )

        self.reward_funcs = reward_funcs
        self.reference_model = self._init_reference_model()
        self.generate_model = self._init_generate_model()

    def _init_reference_model(self):
        reference_model = LlamaModel(self.train_config.llama_config)

        device = 'cpu' # TrainerTools().parallel.device
        reference_model.to(device)
        load_checkpoint_for_eval(model=reference_model, device=device)

        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

        return reference_model

    def _init_generate_model(self):
        generate_model = LlamaModel(self.train_config.llama_config)

        device = 'cpu' #TrainerTools().parallel.device
        generate_model.to(device)
        # load_checkpoint_for_eval(model=generate_model, device=device)

        generate_model.eval()
        for param in generate_model.parameters():
            param.requires_grad = False

        return generate_model

    def _init_loss(self):
        criterion = GRPOLoss(
            clip_eps=self.train_config.grpo_config.clip_eps,
            kl_weight=self.train_config.grpo_config.kl_weight
        )

        return criterion, None

    def _convert_train_args(self) -> Tuple[dict, dict, dict]:
        parallel_kwargs, data_loader_kwargs, sampler_kwargs = super()._convert_train_args()
        data_loader_kwargs.update({
            "batch_size":self.train_config.grpo_config.rollouts_per_step,
            "collate_fn": lambda x: x
        })

        return parallel_kwargs, data_loader_kwargs, sampler_kwargs

    def _create_dataset(self, file_path) -> Dataset:
        return GRPORolloutDataset(file_path)

    def _calc_loss(self, inputs, attention_mask, logits, labels):
        pass

    def _rollout_per_question(
            self,
            prompt_ids: torch.Tensor,
            answer: torch.Tensor
    ):
        pad_token_id = TrainerTools().tokenizer.pad
        device = TrainerTools().parallel.device

        # prompt_ids [prompt_len,]
        prompt_len = prompt_ids.shape[0]
        group_size = self.train_config.grpo_config.group_size

        # [group_size, prompt_len]
        group_prompt_ids = prompt_ids.repeat(group_size, 1)
        # [group_size, prompt_len]
        attention_masks = group_prompt_ids != pad_token_id

        # [group_size, max_seq_len]
        generate_ids = batch_generate(
            # model=self.train_model,
            model=self.generate_model,
            tokens=group_prompt_ids,
            pad_token_id=pad_token_id,
            attention_mask=attention_masks,
            max_position_embeddings=self.train_config.llama_config.max_position_embeddings,
            max_new_tokens=self.train_config.grpo_config.gen_max_new_tokens,
            temperature=self.train_config.grpo_config.gen_temperature,
            k=self.train_config.grpo_config.gen_k,
            p=self.train_config.grpo_config.gen_p,
            device=device,
            suppress_tokens=self.train_config.grpo_config.gen_suppress_tokens
        )

        # [group_size, max_seq_len]
        masks = torch.zeros_like(generate_ids, dtype=torch.bool, device=device)
        # mask prompt
        masks[:, prompt_len:] = True
        # mask pad
        masks[generate_ids == pad_token_id] = False
        # shift right
        # [group_size, max_seq_len - 1]
        masks = masks[:, 1:]

        # shape [group_size, 1]
        rewards = torch.zeros(group_size, 1, dtype=torch.float)

        # apply reward for each response
        for step in range(group_size):
            reward = 0.0
            if self.reward_funcs:
                for reward_fun in self.reward_funcs:
                    reward += reward_fun(prompt_ids, generate_ids[step], answer)

            rewards[step] = reward

        return generate_ids, rewards, masks

    def train(self):
        # 梯度累积步数
        gradient_accumulation_steps = self.train_config.gradient_accumulation_steps
        global_steps = 0
        loss_accumulation = 0.0

        grpo_dataset = GRPODataset()

        aux_loss_coef = self.train_config.loss_config.aux_loss_coef
        file_count = len(self.train_config.all_files)

        for file_idx in range(file_count):
            file_path = self.train_config.all_files[file_idx]
            dataset = self._create_dataset(file_path)

            train_data_loader = TrainerTools().parallel.process_dataloader(
                dataset=dataset,
                data_loader_kwargs=self.data_loader_kwargs,
                sampler_kwargs=self.sampler_kwargs
            )
            batch_count_per_file = len(train_data_loader)

            TrainerTools().parallel.on_epoch_start(file_idx)
            self._on_file_start(-1, file_path)

            for real_batch, batch_data in enumerate(train_data_loader):
                global_steps += 1
                if global_steps < self.last_global_steps:
                    continue

                grpo_dataset.clear()
                # self.train_model.eval()

                # 使用单独的模型生成数据， 原因是在deepspeed并行训练时，使用train_model生成数据会卡死
                self.generate_model.to(TrainerTools().parallel.device)
                self.reference_model.to(TrainerTools().parallel.device)

                # 保存了train_model checkpoint后，这里保证生成模型使用的参数是最新
                if real_batch % self.train_config.grpo_config.load_state_step_interval == 0:
                    log(f'load state for step {real_batch}')
                    load_checkpoint_for_eval(self.generate_model, TrainerTools().parallel.device)

                with torch.inference_mode():
                    for item in batch_data:
                        question = item["question"].to(TrainerTools().parallel.device)
                        answer = item["answer"].to(TrainerTools().parallel.device)

                        # sequence_ids [group_size, max_generate_len]
                        # rewards [group_size, 1]
                        # masks [group_size, max_generate_len - 1]
                        sequence_ids, rewards, masks = self._rollout_per_question(question, answer)

                        advantages = self.criterion.group_advantages(rewards)
                        attention_mask = sequence_ids != TrainerTools().tokenizer.pad

                        # [group_size, max_generate_len - 1]
                        log_probs, _ = self.criterion.sequences_log_probs(
                            model=self.generate_model,
                            sequence_ids=sequence_ids,
                            attention_mask=attention_mask,
                        )

                        # [group_size, max_generate_len - 1]
                        ref_log_probs, _ = self.criterion.sequences_log_probs(
                            model=self.reference_model,
                            sequence_ids=sequence_ids,
                            attention_mask=attention_mask,
                        )

                        rollout_per_batch = {
                            'sequence_ids': sequence_ids,
                            'old_log_probs': log_probs,
                            'ref_log_probs': ref_log_probs,
                            'advantages': advantages,
                            'attention_mask': attention_mask,
                            'mask': masks,
                        }
                        grpo_dataset.append(rollout_per_batch)

                # 卸载到cpu上，等待下次使用时再to gpu
                self.generate_model.to('cpu')
                self.reference_model.to('cpu')
                torch.cuda.empty_cache()

                data_loader = DataLoader(
                    dataset=grpo_dataset,
                    batch_size=self.train_config.batch_size,
                    collate_fn=join_batch,
                    shuffle=True,
                    drop_last=True
                )

                rollout_batch_count = len(data_loader)

                for epoch in range(self.train_config.n_epochs):
                    self.train_model.train()
                    last_ckpt_batch = 0

                    for batch, rollout_data in enumerate(data_loader):
                        # 是否需要更新梯度
                        if gradient_accumulation_steps > 1:
                            need_update_grad = (batch + 1) % gradient_accumulation_steps == 0 or batch == rollout_batch_count - 1
                        else:
                            need_update_grad = True

                        try:
                            sequence_ids = rollout_data['sequence_ids'].to(TrainerTools().parallel.device)
                            old_log_probs = rollout_data['old_log_probs'].to(TrainerTools().parallel.device)
                            ref_log_probs = rollout_data['ref_log_probs'].to(TrainerTools().parallel.device)
                            advantages = rollout_data['advantages'].to(TrainerTools().parallel.device)
                            attention_mask = rollout_data['attention_mask'].to(TrainerTools().parallel.device)
                            mask = rollout_data['mask'].to(TrainerTools().parallel.device)

                            if TrainerTools().parallel.parallel_train:
                                self.train_model.require_backward_grad_sync = need_update_grad

                            with self.ctx:
                                # [batch_size, max_generate_len - 1]
                                log_probs, aux_loss = self.criterion.sequences_log_probs(
                                    model=self.train_model,
                                    sequence_ids=sequence_ids,
                                    attention_mask=attention_mask,
                                )

                                loss, kl = self.criterion(
                                    log_probs=log_probs,
                                    old_log_probs=old_log_probs,
                                    ref_log_probs=ref_log_probs,
                                    mask=mask,
                                    advantages=advantages
                                )

                                if aux_loss_coef and aux_loss:
                                    loss += aux_loss_coef * aux_loss

                            if gradient_accumulation_steps > 1:
                                loss = loss / gradient_accumulation_steps

                            loss_accumulation += loss.detach()
                            self._backward_loss(loss)

                            if need_update_grad:
                                # todo check all_reduce??
                                if TrainerTools().parallel.parallel_train:
                                    dist.all_reduce(loss_accumulation, dist.ReduceOp.AVG)

                                # ds模式已经集成gradient_clipping
                                if not isinstance(TrainerTools().parallel, DsParallel) and self.lr_scheduler.can_clip_grad():
                                    # clip grad
                                    self.scalar.unscale_(self.optimizer)
                                    torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), 1.0)

                                self._step()

                                self._log_loss(
                                    epoch_tag=f'epoch: {epoch}',
                                    file_tag=f'file: {file_idx + 1}/{file_count}',
                                    batch_tag=f'real_batch: {real_batch}/{batch_count_per_file}, batch: {batch}/{rollout_batch_count}',
                                    loss=loss_accumulation.item()
                                )
                                # reset to default
                                loss_accumulation = 0.0
                        except Exception as e:
                            self._on_exception(e, epoch, batch)
                        finally:
                            if need_update_grad and (batch - last_ckpt_batch) >= self.train_config.eval_batch_interval:
                                save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                                save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                                last_ckpt_batch = batch
                                self._on_batch_end(tag=f'epoch:{epoch}/real_bach:{real_batch}/batch:{batch}')
                            try:
                                del loss
                            except UnboundLocalError:
                                pass

                    # end epoch
                    save_checkpoint(model=self.train_model, optimizer=self.optimizer)
                    save_steps(global_steps=global_steps, lr_scheduler=self.lr_scheduler)
                    TrainerTools().parallel.on_epoch_end(epoch)
                    self._on_epoch_end(tag=f'epoch:{epoch}')

        # 等待checkpoint保存完成
        time.sleep(10)
        TrainerTools().parallel.destroy()


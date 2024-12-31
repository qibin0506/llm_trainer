from contextlib import nullcontext
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import pickle

from llama import LlamaModel

from .scheduler import CosineAnnealingWarmupLRScheduler, NoneLRScheduler
from .train_args import TrainArgs
from .parallel_fsdp import FsdpParallel
from .train_tools import TrainerTools
from .checkpoint import load_checkpoint, save_checkpoint
# from .app_state import save_dcp, load_dcp
from .loss import LMLoss, KDLoss
from .utils import (
    set_seed,
    pretrain_padding_fn,
    sft_padding_fn,
)

from .trainer_log import (
    on_batch,
    on_exception,
    on_epoch,
    on_file,
)


class LLMDataset(Dataset):
    def __init__(self, ctx_len, file_path):
        super().__init__()
        self.ctx_len = ctx_len

        self.tokens = []
        with open(file_path, 'rb') as f:
            tokens = pickle.load(f)
            self.tokens.extend(tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        inputs = self.tokens[item]
        inputs = inputs[:self.ctx_len]
        return torch.tensor(inputs).long()

    def get_first(self):
        inputs = self.tokens[0]
        inputs = inputs[:self.ctx_len]
        return torch.tensor(inputs).long()


def train(
        *,
        train_args: TrainArgs,
        prompt_on_batch: str,
        prompt_on_epoch: str,
):
    set_seed()

    if isinstance(TrainerTools().parallel, FsdpParallel) and train_args.fsdp_args is not None:
        fsdp_kwargs = {
            'transformer_layer_cls': train_args.fsdp_args.transformer_layer_cls,
            'wrap_policy_num_params': train_args.fsdp_args.wrap_policy_num_params,
            'cpu_offload': train_args.fsdp_args.cpu_offload,
            'offload_params': train_args.fsdp_args.offload_params
        }
    else:
        fsdp_kwargs = None

    llama_config = train_args.llama_config

    llama = TrainerTools().parallel.process_model(
        LlamaModel(llama_config),
        kwargs=fsdp_kwargs
    )

    if TrainerTools().use_amp:
        ctx = torch.autocast(
            device_type=TrainerTools().parallel.device_type,
            dtype=TrainerTools().dtype,
            enabled=TrainerTools().use_amp
        )
    else:
        ctx = nullcontext()

    if TrainerTools().parallel.is_main_process:
        # eval_model = llama
        eval_model = LlamaModel(llama_config).to('cpu')
    else:
        eval_model = None

    if TrainerTools().parallel.is_main_process:
        total_params = sum(p.numel() for p in llama.parameters())
        print(f"Total number of parameters: {total_params:,}")

        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total size of the model: {total_size_mb:.2f} MB")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scalar = torch.GradScaler(enabled=TrainerTools().use_amp)

    # 梯度累积步数
    gradient_accumulation_steps = train_args.gradient_accumulation_steps
    batch_count = train_args.all_data_size // TrainerTools().parallel.world_size // train_args.batch_size

    print(f"real batch count: {batch_count}")

    if gradient_accumulation_steps > 1:
        batch_count = batch_count // gradient_accumulation_steps

    # 学习率要根据GPU的数量进行倍增：
    # 在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
    initial_lr = train_args.lr_scheduler_args.initial_lr * TrainerTools().parallel.world_size
    optimizer = torch.optim.AdamW(llama.parameters(), lr=initial_lr, weight_decay=0.1)

    if train_args.lr_scheduler_args.enable_lr_scheduler:
        train_iters = batch_count * train_args.n_epochs
        warmup_iters = int(train_args.lr_scheduler_args.warmup_iters_ratio * train_iters)
        min_lr = train_args.lr_scheduler_args.min_lr_ratio * initial_lr
        max_lr = train_args.lr_scheduler_args.max_lr * TrainerTools().parallel.world_size

        lr_scheduler = CosineAnnealingWarmupLRScheduler(
            optimizer=optimizer,
            warmup_iters=warmup_iters,
            initial_lr=initial_lr,
            min_lr=min_lr,
            max_lr=max_lr,
            total_iters=train_iters
        )
    else:
        lr_scheduler = NoneLRScheduler()

    criterion = LMLoss()
    kd_loss = KDLoss() if train_args.kd_args is not None else None

    load_checkpoint(
        llama,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=TrainerTools().parallel.device
    )

    dataloader_args = train_args.data_loader_args

    for epoch in range(train_args.n_epochs):
        batch_count_per_epoch = 0
        loss_accumulation = torch.tensor(0.0, device=TrainerTools().parallel.device)
        llama.train()

        for file in train_args.all_files:
            dataset = LLMDataset(llama_config.max_position_embeddings, file)
            data_loader_kwargs = {
                "batch_size": train_args.batch_size,
                "pin_memory": dataloader_args.data_loader_pin_memory,
                "collate_fn": sft_padding_fn if train_args.is_sft else pretrain_padding_fn,
                "num_workers": dataloader_args.data_loader_num_workers,
                "shuffle": dataloader_args.data_loader_shuffle,
                "drop_last": dataloader_args.data_loader_drop_last,
            }
            sampler_kwargs = {
                "shuffle": dataloader_args.data_loader_shuffle,
                "drop_last": dataloader_args.data_loader_drop_last,
            }

            train_data_loader = TrainerTools().parallel.create_dataloader(
                dataset=dataset,
                data_loader_kwargs=data_loader_kwargs,
                sampler_kwargs=sampler_kwargs
            )

            batch_count_per_file = len(train_data_loader)
            batch_count_per_epoch += batch_count_per_file

            TrainerTools().parallel.on_epoch_start(epoch)
            last_ckpt_batch = 0

            for batch, (inputs, labels) in enumerate(train_data_loader):
                # 是否需要更新梯度
                if gradient_accumulation_steps > 1:
                    need_update_grad = (batch + 1) % gradient_accumulation_steps == 0 or batch == batch_count_per_file - 1
                    if TrainerTools().parallel.is_main_process and need_update_grad:
                        print(f"need_update_grad: {batch}")
                else:
                    need_update_grad = True

                try:
                    inputs, labels = inputs.to(TrainerTools().parallel.device), labels.to(TrainerTools().parallel.device)
                    attention_mask = inputs != TrainerTools().tokenizer.pad

                    if TrainerTools().parallel.parallel_train:
                        # in DDP training we only need to sync gradients at the last micro step.
                        # the official way to do this is with model.no_sync() context manager, but
                        # I really dislike that this bloats the code and forces us to repeat code
                        # looking at the source of that context manager, it just toggles this variable
                        llama.require_backward_grad_sync = need_update_grad

                    with ctx:
                        logits, _ = llama(inputs, attention_mask=attention_mask)
                        # calc loss
                        loss = criterion(logits, labels)

                        # 知识蒸馏loss
                        if kd_loss is not None:
                            teacher_logits = train_args.kd_args.teacher_logits_provider(inputs, attention_mask)
                            distil_loss = kd_loss(logits, teacher_logits, labels)
                            loss = (1 - train_args.kd_args.kd_coef) * loss + train_args.kd_args.kd_coef * distil_loss

                        if gradient_accumulation_steps > 1:
                            loss = loss / gradient_accumulation_steps

                    scalar.scale(loss).backward()

                    if need_update_grad:
                        # clip grad
                        scalar.unscale_(optimizer)

                        if lr_scheduler.can_clip_grad():
                            torch.nn.utils.clip_grad_norm_(llama.parameters(), 1.0)

                        lr_scheduler.step()

                        scalar.step(optimizer)
                        # optimizer.step()
                        scalar.update()
                        # flush the gradients as soon as we can, no need for this memory anymore
                        optimizer.zero_grad(set_to_none=True)

                        TrainerTools().parallel.synchronize()

                    loss_accumulation += loss.detach()

                    if need_update_grad and (batch - last_ckpt_batch) >= 100:
                        last_ckpt_batch = batch
                        save_checkpoint(model=llama, optimizer=optimizer, lr_scheduler=lr_scheduler)

                        loss_item = loss.detach().item()
                        on_batch(
                            eval_model,
                            epoch,
                            batch,
                            batch_count_per_file,
                            loss_item * gradient_accumulation_steps if gradient_accumulation_steps > 1 else loss_item,
                            need_update_grad,
                            prompt_on_batch,
                            llama_config.max_position_embeddings
                        )

                    del loss
                except Exception as e:
                    on_exception(e, epoch, batch)

            on_file(
                eval_model,
                epoch,
                TrainerTools().tokenizer.decode_to_text(dataset.get_first()),
                llama_config.max_position_embeddings,
                file
            )

        if TrainerTools().parallel.parallel_train:
            dist.all_reduce(loss_accumulation, dist.ReduceOp.AVG)

        TrainerTools().parallel.on_epoch_end(epoch)
        save_checkpoint(model=llama, optimizer=optimizer, lr_scheduler=lr_scheduler)

        on_epoch(
            eval_model,
            epoch,
            loss_accumulation.detach().item() / batch_count_per_epoch,
            need_update_grad,
            prompt_on_epoch,
            llama_config.max_position_embeddings
        )

    TrainerTools().parallel.destroy()




"""
todo: 
1. 调研fsdp2 没有太多资料
2. 蒸馏 完成
3. Yarn和phi3的Phi3LongRoPEScaledRotaryEmbedding调研
4. MLA调研
5. DPO调研
6. inference使用缓存model，每个进程缓存一个
7. 多模态
"""
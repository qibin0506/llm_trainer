import os
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
import pickle

from pytorch.llm.llama import LlamaModel, LlamaConfig
from pytorch.llm.llm_trainer.scheduler import CosineAnnealingWarmupScheduler
from pytorch.llm.llm_trainer.utils import (
    TrainConfig,
    calc_loss,
    pretrain_padding_fn,
    sft_padding_fn,
)

from pytorch.llm.llm_trainer.trainer_log import (
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
        n_epochs: int,
        batch_size: int,
        *,
        llama_config: LlamaConfig,
        all_data_size: int,
        all_files: list[any],
        is_sft: bool, # 是否sft
        prompt_on_batch: str,
        prompt_on_epoch: str,
):
    llama = TrainConfig().ddp_helper.process_model(LlamaModel(llama_config), f"{os.environ['SAVE_DIR']}modeling.pth")

    if TrainConfig().ddp_helper.is_main_process():
        eval_model = LlamaModel(llama_config).to('cpu')
        if TrainConfig().ddp_helper.use_compile:
            torch.compile(eval_model)
    else:
        eval_model = None

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scalar = torch.GradScaler(enabled=TrainConfig().use_amp)

    if TrainConfig().ddp_helper.is_main_process():
        total_params = sum(p.numel() for p in llama.parameters())
        print(f"Total number of parameters: {total_params:,}")

        total_size_bytes = total_params * 4
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Total size of the model: {total_size_mb:.2f} MB")

    gradient_accumulation_steps = TrainConfig().gradient_accumulation_steps
    batch_count = all_data_size // TrainConfig().ddp_helper.world_size() // batch_size

    if gradient_accumulation_steps > 1:
        batch_count = batch_count // gradient_accumulation_steps

    train_iters = batch_count * n_epochs

    warmup_iters = int(0.2 * train_iters)
    # 学习率要根据GPU的数量进行倍增：
    # 在训练的过程中，损失梯度决定下降的方向，学习率决定下降的步长。如果有两块gpu，前进的综合步长为：平均学习率*2
    initial_lr = 1e-5 * TrainConfig().ddp_helper.world_size()
    min_lr = 0.1 * initial_lr
    max_lr = 5e-4 * TrainConfig().ddp_helper.world_size()

    optimizer = torch.optim.AdamW(llama.parameters(), lr=initial_lr, weight_decay=0.1)
    lr_scheduler = CosineAnnealingWarmupScheduler(optimizer, warmup_iters, initial_lr, min_lr, max_lr, train_iters)

    for epoch in range(n_epochs):
        loss_accumulation = torch.tensor(0.0, device=TrainConfig().ddp_helper.device)
        llama.train()

        for file in all_files:
            dataset = LLMDataset(llama_config.max_position_embeddings, file)
            train_data_loader = TrainConfig().ddp_helper.create_dataloader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=sft_padding_fn if is_sft else pretrain_padding_fn
            )

            train_loader_len = len(train_data_loader)
            TrainConfig().ddp_helper.on_epoch(epoch)

            for batch, (inputs, labels) in enumerate(train_data_loader):
                # 是否需要更新梯度
                if gradient_accumulation_steps > 1:
                    need_update_grad = (batch + 1) % gradient_accumulation_steps == 0 or batch == train_loader_len - 1
                    if TrainConfig().ddp_helper.is_main_process() and need_update_grad:
                        print(f"need_update_grad: {batch}")
                else:
                    need_update_grad = True

                try:
                    inputs, labels = inputs.to(TrainConfig().ddp_helper.device), labels.to(TrainConfig().ddp_helper.device)
                    attention_mask = inputs != TrainConfig().tokenizer.pad

                    if TrainConfig().ddp_helper.ddp:
                        # in DDP training we only need to sync gradients at the last micro step.
                        # the official way to do this is with model.no_sync() context manager, but
                        # I really dislike that this bloats the code and forces us to repeat code
                        # looking at the source of that context manager, it just toggles this variable
                        llama.require_backward_grad_sync = need_update_grad

                    with torch.autocast(device_type=TrainConfig().ddp_helper.device_type,
                                        dtype=TrainConfig().dtype, enabled=TrainConfig().use_amp):
                        logits, _ = llama(inputs, attention_mask=attention_mask)
                        # calc loss
                        loss = calc_loss(logits, labels)

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

                        TrainConfig().ddp_helper.synchronize()

                    loss_accumulation += loss.detach()

                    if gradient_accumulation_steps > 1:
                        on_batch(eval_model, epoch, batch, loss.detach().item() * gradient_accumulation_steps,
                                 need_update_grad, prompt_on_batch)
                    else:
                        on_batch(eval_model, epoch, batch, loss.detach().item(), need_update_grad, prompt_on_batch)

                    del loss
                except KeyboardInterrupt as e:
                    on_exception(e, epoch, batch)
                except Exception as e:
                    on_exception(e, epoch, batch)

            on_file(eval_model, epoch, batch, TrainConfig().tokenizer.decode_to_text(dataset.get_first()))

        if TrainConfig().ddp_helper.ddp:
            dist.all_reduce(loss_accumulation, dist.ReduceOp.AVG)

        TrainConfig().ddp_helper.end_epoch(epoch)
        on_epoch(eval_model, epoch, loss_accumulation, batch_count, need_update_grad, prompt_on_epoch)

    TrainConfig().ddp_helper.destroy()

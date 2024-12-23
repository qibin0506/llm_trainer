import os
import threading
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig
)

from llama import LlamaConfig
from .generate_utils import generate
from .parallel_fsdp import FsdpParallel
from .utils import TrainerTools


def _submit_gen_task(eval_model, tag, state_dict, prompt, max_position_embeddings, max_new_tokens):
    def task():
        save_dir = os.environ['SAVE_DIR']

        with open(f'{save_dir}gen_temp.txt', 'w') as tf:
            def write_temp(item):
                tf.write(item)
                tf.flush()

            eval_model.load_state_dict(state_dict)
            gen = generate(
                eval_model,
                prompt=prompt,
                max_position_embeddings=max_position_embeddings,
                max_new_tokens=max_new_tokens,
                temperature=None,
                k=None,
                p=None,
                device='cpu',
                item_callback=lambda item: write_temp(item)
            )

        with open(f'{save_dir}gen.txt', 'a') as f:
            f.write(f"{tag}, gen->{gen}\n")

    threading.Thread(target=task).start()


def _save_model(state_dict):
    save_dir = os.environ['SAVE_DIR']
    if os.path.exists(f'{save_dir}modeling.pth'):
        if os.path.exists(f'{save_dir}modeling_bak.pth'):
            os.remove(f'{save_dir}modeling_bak.pth')

        os.rename(f'{save_dir}modeling.pth', f'{save_dir}modeling_bak.pth')

    ckpt = {'model': state_dict}

    # if isinstance(TrainerTools().parallel, FsdpParallel):
    #     # 如果是fsdp模式，则需要等所有gpu本轮都完毕后才能保存
    #     # use a barrier to make sure training is done on all ranks
    #     dist.barrier()
    #
    #     ckpt = {'model': TrainerTools().parallel.model.state_dict()}
    # else:
    #     ckpt = {'model': TrainerTools().parallel.raw_model.state_dict()}

    torch.save(ckpt, f'{save_dir}modeling.pth')


def _get_model_state_dict():
    if isinstance(TrainerTools().parallel, FsdpParallel):
        fsdp_model = TrainerTools().parallel.model
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp_model, 'FULL_STATE_DICT', full_state_dict_config):
            state_dict = fsdp_model.state_dict()
    else:
        state_dict = TrainerTools().parallel.raw_model.state_dict()

    return state_dict


def on_batch(eval_model, epoch, batch, loss, need_update_grad, prompt, llama_config: LlamaConfig):
    if TrainerTools().parallel.is_main_process:
        if (batch + 1) % 100 == 0:
            save_dir = os.environ['SAVE_DIR']
            print(f"epoch: {epoch}, batch: {batch}")
            with open(f'{save_dir}batch.txt', 'a') as f:
                f.write(f"epoch: {epoch}, batch: {batch}, loss: {loss}, need_update_grad: {need_update_grad}\n")

            state_dict = _get_model_state_dict()

            _save_model(state_dict)
            _submit_gen_task(
                eval_model,
                tag=f'sign:batch/epoch:{epoch}/batch:{batch}',
                state_dict=state_dict,
                prompt=prompt,
                max_position_embeddings=llama_config.max_position_embeddings,
                max_new_tokens=246
            )


def on_file(eval_model, epoch, batch, prompt, llama_config: LlamaConfig):
    if TrainerTools().parallel.is_main_process:
        state_dict = _get_model_state_dict()
        _submit_gen_task(
            eval_model,
            tag=f'sign:file/epoch:{epoch}/batch:{batch}',
            state_dict=state_dict,
            prompt=prompt,
            max_position_embeddings=llama_config.max_position_embeddings,
            max_new_tokens=246
        )


def on_exception(e, epoch, batch):
    if isinstance(e, KeyboardInterrupt):
        if TrainerTools().parallel.is_main_process:
            # ddp_helper.raw_model.train()
            # save_model()
            exit(0)
    elif isinstance(e, Exception):
        if TrainerTools().parallel.is_main_process:
            save_dir = os.environ['SAVE_DIR']
            with open(f'{save_dir}batch.txt', 'a') as f:
                exception_file = e.__traceback__.tb_frame.f_globals["__file__"]
                exception_line = e.__traceback__.tb_lineno
                f.write(f"epoch: {epoch}, batch: {batch}, {e} at {exception_file} line {exception_line}\n")


def on_epoch(eval_model, epoch, loss_accumulation, all_batch, need_update_grad, prompt, llama_config: LlamaConfig):
    if TrainerTools().parallel.is_main_process:
        state_dict = _get_model_state_dict()
        _save_model(state_dict)

        # test_loss = test_loop(model, test_data_loader)
        print(f'train_loss: {loss_accumulation.detach().item() / all_batch}')

        save_dir = os.environ['SAVE_DIR']
        with open(f'{save_dir}batch.txt', 'a') as f:
            f.write(
                f"epoch: {epoch},"
                f" loss: {loss_accumulation.detach().item() / all_batch},"
                f" need_update_grad:{need_update_grad}\n")

        _submit_gen_task(
            eval_model,
            tag=f'sign:epoch/epoch:{epoch}',
            state_dict=state_dict,
            prompt=prompt,
            max_position_embeddings=llama_config.max_position_embeddings,
            max_new_tokens=246
        )
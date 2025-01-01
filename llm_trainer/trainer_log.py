import os
import threading
import time
from multiprocessing import Process
import torch

from .generate_utils import generate
from .train_tools import TrainerTools
from .checkpoint import load_checkpoint
from .utils import log


def _get_save_dir() -> str:
    save_dir = os.environ['SAVE_DIR']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    return save_dir

def _eval_task(eval_model, tag, prompt, max_position_embeddings, is_new_process):
    save_dir = _get_save_dir()
    ident = os.getpid() if is_new_process else 'thread'

    with open(f'{save_dir}gen_temp_{ident}.txt', 'w') as tf:
        def write_temp(item):
            tf.write(item)
            tf.flush()

        # 当eval_model不是独立model时可以尝试这个
        # if isinstance(eval_model, FSDP):
        #     with FSDP.summon_full_params(module=eval_model, writeback=False, recurse=False):
        #         gen = generate(
        #             eval_model,
        #             prompt=prompt,
        #             max_position_embeddings=max_position_embeddings,
        #             max_new_tokens=max_new_tokens,
        #             # temperature=None,
        #             # k=None,
        #             # p=None,
        #             device='cpu',
        #             item_callback=lambda item: write_temp(item)
        #         )

        # ---------

        load_checkpoint(eval_model, device='cpu')

        gen = generate(
            eval_model,
            prompt=prompt,
            max_position_embeddings=max_position_embeddings,
            max_new_tokens=100,
            temperature=0.6,
            k=3,
            p=None,
            device='cpu',
            item_callback=lambda item: write_temp(item)
        )

    with open(f'{save_dir}gen.txt', 'a') as f:
        f.write(f"{tag}, gen->{gen}\n")


def _submit_gen_task(eval_model: torch.nn.Module, tag, prompt, max_position_embeddings):
    threading.Thread(target=_eval_task, args=(eval_model, tag, prompt, max_position_embeddings, False)).start()
    # Process(target=_eval_task, args=(eval_model, tag, prompt, max_position_embeddings, True)).start()


def on_batch(
        eval_model,
        epoch,
        batch,
        batch_count,
        loss,
        need_update_grad,
        prompt,
        max_position_embeddings
):
    if TrainerTools().parallel.is_main_process:
        save_dir = _get_save_dir()
        log(f"epoch: {epoch}, batch: {batch}/{batch_count}")
        log(
            f"epoch: {epoch}, batch: {batch}/{batch_count}, loss: {loss}, need_update_grad: {need_update_grad}\n",
            f'{save_dir}batch.txt'
        )

        _submit_gen_task(
            eval_model,
            tag=f'sign:batch/epoch:{epoch}/batch:{batch}',
            prompt=prompt,
            max_position_embeddings=max_position_embeddings
        )


def on_file(
        eval_model,
        epoch,
        prompt,
        max_position_embeddings,
        file_name
):
    if TrainerTools().parallel.is_main_process:
        _submit_gen_task(
            eval_model,
            tag=f'sign:file/epoch:{epoch}',
            prompt=prompt,
            max_position_embeddings=max_position_embeddings
        )

        log(f"epoch: {epoch}, {file_name} train finish.\n", f'{_get_save_dir()}batch.txt')

def on_exception(e, epoch, batch):
    if isinstance(e, torch.OutOfMemoryError):
        if TrainerTools().parallel.is_main_process:
            save_dir = _get_save_dir()
            exception_file = e.__traceback__.tb_frame.f_globals["__file__"]
            exception_line = e.__traceback__.tb_lineno
            log(
                f"epoch: {epoch}, batch: {batch}, {e} at {exception_file} line {exception_line}\n",
                f'{save_dir}batch.txt'
            )
    else:
        raise e


def on_epoch(
        eval_model,
        epoch,
        loss,
        need_update_grad,
        prompt,
        max_position_embeddings
):
    if TrainerTools().parallel.is_main_process:

        # test_loss = test_loop(model, test_data_loader)
        save_dir = _get_save_dir()
        log(f'train_loss: {loss}')
        log(
            f"(epoch: {epoch}, loss: {loss}, need_update_grad:{need_update_grad}\n",
            f'{save_dir}batch.txt'
        )

        _submit_gen_task(
            eval_model,
            tag=f'sign:epoch/epoch:{epoch}',
            prompt=prompt,
            max_position_embeddings=max_position_embeddings
        )
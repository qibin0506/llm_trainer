import os
import threading
import time

import torch

from .generate_utils import generate
from .tools import TrainerTools
from .checkpoint import load_checkpoint_for_eval
from .log import log


def _get_log_dir() -> str:
    log_dir = os.environ['LOG_DIR']
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    return log_dir

def _eval_task(eval_model, tag, prompt, max_position_embeddings, is_new_process):
    log_dir = _get_log_dir()

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

    load_checkpoint_for_eval(eval_model, device='cpu')

    gen_result = generate(
        eval_model,
        prompt=prompt,
        max_position_embeddings=max_position_embeddings,
        max_new_tokens=100,
        temperature=0.6,
        k=3,
        p=None,
        device='cpu'
    )

    with open(f'{log_dir}gen.txt', 'a') as f:
        f.write(f"{tag}, gen->{gen_result}\n")


def _submit_gen_task(eval_model: torch.nn.Module, tag, prompt, max_position_embeddings):
    # 等待2s，防止deepspeed模式下，找不到checkpoint问题
    time.sleep(2)
    threading.Thread(target=_eval_task, args=(eval_model, tag, prompt, max_position_embeddings, False)).start()
    # Process(target=_eval_task, args=(eval_model, tag, prompt, max_position_embeddings, True)).start()


def log_loss(
        epoch: int,
        batch: int,
        batch_count: int,
        loss
):
    if TrainerTools().parallel.is_main_process:
        log_dir = _get_log_dir()
        log(f"epoch: {epoch}, batch: {batch}/{batch_count}, loss: {loss}")
        log(
            f"epoch: {epoch}, batch: {batch}/{batch_count}, loss: {loss}\n",
            f'{log_dir}log.txt'
        )

def on_batch_end(
        eval_model,
        epoch,
        batch,
        prompt,
        max_position_embeddings
):
    if TrainerTools().parallel.is_main_process:
        _submit_gen_task(
            eval_model,
            tag=f'sign:batch/epoch:{epoch}/batch:{batch}',
            prompt=prompt,
            max_position_embeddings=max_position_embeddings
        )


def on_file_start(epoch, file_name):
    if TrainerTools().parallel.is_main_process:
        log(f"epoch: {epoch}, start train {file_name}\n", f'{_get_log_dir()}log.txt')


def on_exception(e, epoch, batch):
    if isinstance(e, torch.OutOfMemoryError):
        log_dir = _get_log_dir()
        exception_file = e.__traceback__.tb_frame.f_globals["__file__"]
        exception_line = e.__traceback__.tb_lineno
        log(
            f"epoch: {epoch}, batch: {batch}, {e} at {exception_file} line {exception_line}\n",
            f'{log_dir}log.txt'
        )

        time.sleep(1)

    raise e


def on_epoch_end(
        eval_model,
        epoch,
        prompt,
        max_position_embeddings
):
    if TrainerTools().parallel.is_main_process:
        _submit_gen_task(
            eval_model,
            tag=f'sign:epoch/epoch:{epoch}',
            prompt=prompt,
            max_position_embeddings=max_position_embeddings
        )
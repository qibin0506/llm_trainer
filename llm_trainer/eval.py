import threading
import time

import torch

from .generate_utils import generate
from .checkpoint import load_checkpoint_for_eval
from .utils import get_log_dir


def _eval_task(eval_model, tag, prompt, max_position_embeddings, is_new_process):
    log_dir = get_log_dir()

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
        max_new_tokens=max_position_embeddings,
        temperature=0.6,
        k=3,
        p=None,
        device='cpu'
    )

    with open(f'{log_dir}gen.txt', 'a') as f:
        f.write(f"{tag}, gen->{gen_result}\n")


def submit_gen_task(eval_model: torch.nn.Module, tag, prompt, max_position_embeddings):
    # 等待5s，防止deepspeed模式下，找不到checkpoint问题
    time.sleep(5)
    threading.Thread(target=_eval_task, args=(eval_model, tag, prompt, max_position_embeddings, False)).start()
    # Process(target=_eval_task, args=(eval_model, tag, prompt, max_position_embeddings, True)).start()
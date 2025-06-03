import time

import torch

from .generate_utils import generate
from .checkpoint import load_checkpoint_for_eval
from .log import get_log_dir
from .tools import TrainerTools
from .train_configs import EvalConfig


def _eval_task(
        eval_model: torch.nn.Module,
        eval_config: EvalConfig,
        tag,
        prompt,
        pixel_values,
        max_position_embeddings,
        tokens_per_image,
        device
):
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
    try:
        load_checkpoint_for_eval(eval_model, device=device)
    except:
        return

    gen_result = generate(
        eval_model,
        prompt=prompt,
        max_position_embeddings=max_position_embeddings,
        max_new_tokens=eval_config.max_new_tokens,
        temperature=eval_config.temperature,
        k=eval_config.top_k,
        p=eval_config.top_p,
        pixel_values=pixel_values,
        tokens_per_image=tokens_per_image,
        device=device
    )

    with open(f'{log_dir}gen.txt', 'a') as f:
        f.write(f"{tag}, gen->{gen_result}\n")


def submit_gen_task(
        eval_model: torch.nn.Module,
        eval_config: EvalConfig,
        tag,
        prompt,
        pixel_values,
        max_position_embeddings,
        tokens_per_image
):
    # 等待1s，防止deepspeed模式下，找不到checkpoint问题
    time.sleep(1)
    eval_model.to(TrainerTools().parallel.device)
    _eval_task(
        eval_model=eval_model,
        eval_config=eval_config,
        tag=tag,
        prompt=prompt,
        pixel_values=pixel_values,
        max_position_embeddings=max_position_embeddings,
        tokens_per_image=tokens_per_image,
        device=TrainerTools().parallel.device
    )
    eval_model.to('cpu')

    # threading.Thread(target=_eval_task, args=args).start()
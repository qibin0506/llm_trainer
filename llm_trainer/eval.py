import os
import torch

from .generate_utils import generate
from .tools import TrainerTools
from .train_configs import TrainConfig
from .log import _get_log_dir

def submit_gen_task(
        eval_model: torch.nn.Module,
        train_config: TrainConfig,
        tag,
        prompt,
        pixel_values,
        tokens_per_image
):
    tokens = TrainerTools().tokenizer.encode(prompt, unsqueeze=True, covert_tensor=True)
    max_new_tokens = train_config.eval_config.max_new_tokens

    gen_result = generate(
        eval_model,
        prompt=tokens,
        max_new_tokens=max_new_tokens,
        temperature=train_config.eval_config.temperature,
        k=train_config.eval_config.top_k,
        p=train_config.eval_config.top_p,
        pixel_values=pixel_values,
        tokens_per_image=tokens_per_image,
        device=TrainerTools().parallel.device
    )

    with open(os.path.join(_get_log_dir(), 'gen.txt'), 'a') as f:
        f.write(f"{tag}, gen->{gen_result}\n")

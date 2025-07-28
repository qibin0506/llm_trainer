import torch

from .generate_utils import generate
from .log import get_log_dir
from .tools import TrainerTools
from .train_configs import EvalConfig

def submit_gen_task(
        eval_model: torch.nn.Module,
        eval_config: EvalConfig,
        tag,
        prompt,
        pixel_values,
        max_position_embeddings,
        tokens_per_image
):
    log_dir = get_log_dir()
    tokens = TrainerTools().tokenizer.encode(prompt, unsqueeze=True, covert_tensor=True)

    max_new_tokens = eval_config.max_new_tokens
    if not max_new_tokens:
        max_new_tokens = max_position_embeddings - tokens.shape[-1]

    gen_result = generate(
        eval_model,
        prompt=tokens,
        max_position_embeddings=max_position_embeddings,
        max_new_tokens=max_new_tokens,
        temperature=eval_config.temperature,
        k=eval_config.top_k,
        p=eval_config.top_p,
        pixel_values=pixel_values,
        tokens_per_image=tokens_per_image,
        device=TrainerTools().parallel.device
    )

    with open(f'{log_dir}gen.txt', 'a') as f:
        f.write(f"{tag}, gen->{gen_result}\n")

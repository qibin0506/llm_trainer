# llm_trainer

```python
import torch
from llm_trainer import TrainerTools, FileDataset, train_configs
from llm_model import ModelConfig, RoPEConfig, MoEConfig
import os

class ListFileDataset(FileDataset):
    def __init__(self, files):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> str:
        return self.files[idx]


def init_env():
    #  Of the allocated memory 33.98 GiB is allocated by PyTorch,
    #  and 8.89 GiB is reserved by PyTorch but unallocated.
    #  If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
    #  See documentation for Memory Management
    #  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKENIZERS_TYPE'] = 'zh_llama'  # or qwen
    os.environ['TOKEN_DIR'] = '../llm_model_tokens/'

    os.environ['LOG_DIR'] = './log/'

    os.environ['ENABLE_DCP'] = '1'
    os.environ['DIST_CHECKPOINT_DIR'] = 'ckpt_dir'
    os.environ['CHECKPOINT_NAME'] = 'ckpt.pth'
    os.environ['EVAL_CHECKPOINT_NAME'] = 'eval_ckpt.pth'

    # os.environ['DTYPE'] = 'float32'


def get_model_config():
    return ModelConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=1024,
        moe_n_dense_layer=1,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        attention_implementation='auto',
        rope_config=RoPEConfig(
            rope_theta=1e6
        ),
        moe_config=MoEConfig(
            num_experts_per_tok=2,
            n_routed_experts=8,
            n_shared_experts=1,
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True
        )
    )


def get_vision_tower():
    model: torch.nn.Module = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
    model.to(device=TrainerTools().parallel.device, dtype=TrainerTools().dtype)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    def vision_tower(pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            # (1, 196, 768)
            last_hidden_state = outputs.last_hidden_state.to(pixel_values.dtype)

        return last_hidden_state

    return vision_tower


def get_vlm_config():
    assert int(image_size // patch_size) == 14
    assert int(image_size // patch_size) // int(tokens_per_image**0.5) > 0

    return VLMConfig(
        image_tok=TrainerTools().tokenizer.image,
        image_size=image_size,
        patch_size=patch_size,
        tokens_per_image=tokens_per_image,
        vision_hidden_size=768,
        vision_tower=get_vision_tower(),
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=2048,
        moe_intermediate_size=1024,
        moe_n_dense_layer=1,
        num_hidden_layers=24,
        num_attention_heads=12,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        attention_implementation='auto',
        rope_config=RoPEConfig(
            rope_theta=1e6
        ),
        moe_config=MoEConfig(
            num_experts_per_tok=2,
            n_routed_experts=8,
            n_shared_experts=1,
            aux_loss_alpha=0.1,
            seq_aux=True,
            norm_topk_prob=True
        )
    )

def _get_train_config(
        n_epochs: int,
        train_reasoning_model: bool,
        is_sft: bool,
        is_dpo: bool,
        is_grpo: bool,
        real_batch_size: int,
        file_dataset: FileDataset,
        model_config: ModelConfig
):
    desire_batch_size = real_batch_size * 3
    gradient_accumulation_steps = desire_batch_size // real_batch_size
    eval_batch_interval = 10 if is_grpo else 100

    ds_config = train_configs.DsConfig(zero_config=train_configs.DsZero3Config())

    loss_config = train_configs.LossConfig(
        critical_tokens=[
            TrainerTools().tokenizer.reasoning_start,
            TrainerTools().tokenizer.reasoning_end,
            TrainerTools().tokenizer.answer_start,
            TrainerTools().tokenizer.answer_end
        ],
        critical_alpha=10.0
    ) if train_reasoning_model else train_configs.LossConfig()

    dpo_config = train_configs.DPOConfig(
        loss_beta=0.1,
        loss_label_smoothing=0.0,
        nll_loss_coef=0.2
    ) if is_dpo else None

    grpo_config = train_configs.GRPOConfig(
        grpo_steps=1,
        clip_eps=0.1,
        kl_weight=0.04,
        group_size=16,
        gen_max_new_tokens=500,
        gen_temperature=0.7,
        gen_k=10,
        gen_p=0.5,
        gen_suppress_tokens=None,
    ) if is_grpo else None

    lr_mul = TrainerTools().parallel.world_size
    min_lr_ratio = 0.1

    if is_grpo:
        # grpo all_data_size=8792
        #   train_batch_per_world=epochs*(all_data_size/batch_size/world_size)*grpo_steps
        #       =1*(8792/2/4)*1=1099
        initial_lr = 5e-6 * lr_mul
        max_lr = 1e-5 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 100
        period_mul = 1
        warmup_iters = 100
    elif is_dpo:
        # dpo all_data_size=207339
        # train_batch_per_world=epochs*all_data_size/batch_size/world_size/gradient_accumulation_steps
        #   =2*207339/24/4/3=1439
        initial_lr = 1e-8 * lr_mul
        max_lr = 5e-8 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 100
        period_mul = 1
        warmup_iters = 200
    elif train_reasoning_model:
        # reasoning_size=191917
        # train_batch_per_world=epochs*all_data_size/batch_size/world_size/gradient_accumulation_steps
        #   =2*191917/24/4/3=1332
        initial_lr = 1e-5 * lr_mul
        max_lr = 5e-5 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 200
        period_mul = 1
        warmup_iters = 100
    elif is_sft:
        # sft_1024_size=2274622
        # train_batch_per_world=epochs*all_data_size/batch_size/world_size/gradient_accumulation_steps
        #   =5*2274622/24/4/3=39489
        initial_lr = 1e-5 * lr_mul
        max_lr = 5e-5 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 1000
        period_mul = 1
        warmup_iters = 1000
    else:
        initial_lr = 1e-4 * lr_mul
        max_lr = 5e-4 * lr_mul
        min_lr = initial_lr * min_lr_ratio
        period = 5000
        period_mul = 1
        warmup_iters = 3000

    lr_config = train_configs.LrConfig(
        enable_lr_scheduler=True,
        initial_lr=initial_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        period=period,
        period_mul=period_mul,
        warmup_iters=warmup_iters
    )

    data_loader_config = train_configs.DataLoaderConfig(
        data_loader_pin_memory=True,
        data_loader_num_workers=0,
        data_loader_shuffle=False,
        data_loader_drop_last=True
    )

    train_config = train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_batch_interval=eval_batch_interval,
        loss_config=loss_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        lr_config=lr_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        kd_config=None
    )

    return train_config


def get_pretrain_config():
    pretrain_data_list = [
        './data/deepctrl_long_0.pkl',
        './data/deepctrl_long_1.pkl',
        './data/deepctrl_long_2.pkl',
        './data/deepctrl_long_3.pkl',
        './data/deepctrl_long_4.pkl',
        './data/deepctrl_long_final.pkl',
        './data/deepctrl_short_0.pkl',
        './data/deepctrl_short_1.pkl',
        './data/deepctrl_short_final.pkl',
    ]

    return _get_train_config(
        n_epochs=1,
        train_reasoning_model=False,
        is_sft=False,
        is_dpo=False,
        is_grpo=False,
        real_batch_size=14,
        file_dataset=ListFileDataset(pretrain_data_list),
        model_config=get_model_config()
    )


def get_sft_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_sft=True,
        is_dpo=False,
        is_grpo=False,
        real_batch_size=12,
        file_dataset=ListFileDataset(['./data/sft_deepctrl_short.pkl']),
        model_config=get_model_config()
    )


def get_dpo_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_sft=False,
        is_dpo=True,
        is_grpo=False,
        real_batch_size=6,
        file_dataset=ListFileDataset(['./data/dpo.pkl']),
        model_config=get_model_config()
    )


def get_reasoning_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=True,
        is_dpo=False,
        is_sft=True,
        is_grpo=False,
        real_batch_size=12,
        file_dataset=ListFileDataset(['./data/r1_mix_1024.pkl']),
        model_config=get_model_config()
    )


def get_grpo_config():
    return _get_train_config(
        n_epochs=2,
        train_reasoning_model=False,
        is_dpo=False,
        is_sft=False,
        is_grpo=True,
        real_batch_size=4,
        file_dataset=ListFileDataset(['./data/grpo.pkl']),
        model_config=get_model_config()
    )


```

``` python
from llm_trainer import Trainer
from utils import init_env, get_pretrain_config

# def kd_teacher_logits_provider(inputs, attention_mask):
#     pass

# torchrun --standalone --nproc_per_node=gpu pretrain.py
if __name__ == '__main__':
    init_env()
    eval_prompts = [
        '请问今天北京天气如何？',
        '告诉我世界上最大的湖是哪个？',
        '介绍一下上海'
    ]

    trainer = Trainer(
        train_config=get_pretrain_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()
```

``` python
from llm_trainer import SFTTrainer
from utils import init_env, get_sft_config
from constant import system_prompt

# def kd_teacher_logits_provider(inputs, attention_mask):
#     pass

# torchrun --standalone --nproc_per_node=gpu pretrain.py
if __name__ == '__main__':
    init_env()

    # <system>{system_prompt}</s><user>你好</s><assistant>我好</s><user>很好</s><assistant>不好</s>

    eval_prompts = [
        f'{system_prompt}<user>告诉我世界上最大的湖是哪个？</s><assistant>',
        f'{system_prompt}<user>请问今天北京天气如何？</s><assistant>',
        f'{system_prompt}<user>哪吒和孙悟空谁更厉害？</s><assistant>',
        f'{system_prompt}<user>保持健康的三个提示是什么？</s><assistant>',
        f'{system_prompt}<user>你是谁？</s><assistant>'
    ]

    trainer = SFTTrainer(
        train_config=get_sft_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()
```

``` python
from llm_trainer import DPOTrainer
from utils import init_env, get_dpo_config
from constant import system_prompt

# def kd_teacher_logits_provider(inputs, attention_mask):
#     pass

# torchrun --standalone --nproc_per_node=gpu pretrain.py
if __name__ == '__main__':
    init_env()

    # <system>{system_prompt}</s><user>你好</s><assistant>我好</s><user>很好</s><assistant>不好</s>

    eval_prompts = [
        f'{system_prompt}<user>告诉我世界上最大的湖是哪个？</s><assistant>',
        f'{system_prompt}<user>请问今天北京天气如何？</s><assistant>',
        f'{system_prompt}<user>哪吒和孙悟空谁更厉害？</s><assistant>',
        f'{system_prompt}<user>保持健康的三个提示是什么？</s><assistant>',
        f'{system_prompt}<user>你是谁？</s><assistant>'
    ]

    trainer = DPOTrainer(
        train_config=get_dpo_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()
```

``` python
from llm_trainer import SFTTrainer
from utils import init_env, get_reasoning_config
from constant import reasoning_system_prompt

# def kd_teacher_logits_provider(inputs, attention_mask):
#     pass

# torchrun --standalone --nproc_per_node=gpu pretrain.py
if __name__ == '__main__':
    init_env()

    # <system>{reasoning_system_prompt}</s><user>你好</s><assistant><reasoning>思考</reasoning><answer>回答</answer></s>

    eval_prompts = [
        f'{reasoning_system_prompt}<user>告诉我世界上最大的湖是哪个？</s><assistant>',
        f'{reasoning_system_prompt}<user>请问今天北京天气如何？</s><assistant>',
        f'{reasoning_system_prompt}<user>哪吒和孙悟空谁更厉害？</s><assistant>',
        f'{reasoning_system_prompt}<user>保持健康的三个提示是什么？</s><assistant>',
        f'{reasoning_system_prompt}<user>你是谁？</s><assistant>'
    ]

    trainer = SFTTrainer(
        train_config=get_reasoning_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()
```

``` python
from typing import List, Optional
import re
import torch
from llm_trainer import GRPOTrainer, TrainerTools
from utils import init_env, get_grpo_config
from constant import reasoning_system_prompt


# todo 提取<answer></answer>
def extract_answer_from_completion(completion_text: str)-> str:
    # <reasoning>思考</reasoning><answer>回答</answer></s>
    parts = completion_text.split("<answer>")
    if len(parts) < 2:
        return ''

    # 回答</answer></s>
    last_part = parts[-1]

    # Extract content up to </answer>
    if "</answer>" not in last_part:
        return ''

    # 回答
    answer = last_part.split("</answer>")[0].strip()
    return '' if answer == "..." else answer


def get_last_number(response_answer: str)-> Optional[str]:
    numbers = re.findall(r'-?\d+\.?\d*', response_answer)
    if numbers:
        last_num = numbers[-1]
        return last_num

    return None


def get_reward(completion_text: str, correct_answer: str)-> float:
    reward = 0.0
    response_answer = extract_answer_from_completion(completion_text)
    response_last_number = get_last_number(response_answer)

    #  正确答案奖励: 0.0 ~ 2.0
    if response_last_number == correct_answer:
        reward += 2.0 # 答案相同，奖励2
    elif correct_answer in response_answer:
        reward += 1.5 # 正确答案在回答中，奖励1.5

    #  回答格式奖励: 0.0 ~ 1.0
    if TrainerTools().tokenizer.text_reasoning_start in completion_text:
        reward += 0.2 # 答案包含<reasoning>，奖励0.2

    if TrainerTools().tokenizer.text_reasoning_end in completion_text:
        reward += 0.2 # 答案包含</reasoning>，奖励0.2

    if TrainerTools().tokenizer.text_answer_start in completion_text:
        reward += 0.2 # 答案包含<answer>，奖励0.2

    if TrainerTools().tokenizer.text_answer_end in completion_text:
        reward += 0.2 # 答案包含</answer>，奖励0.2

    if TrainerTools().tokenizer.text_end in completion_text:
        reward += 0.2 # 答案包含</s>，奖励0.2

    # 总奖励：0.0 ~ 3.0
    return reward


def reward_func(prompt_ids: torch.Tensor, completion_ids: torch.Tensor, answers: torch.Tensor) -> List[float]:
    # 1. 如果回答包含思考部分，则奖励1.25分
    # 2. 如果正确答案相同，则奖励1分
    # 3. 如果正确答案在回答中，则奖励0.5分

    rewards = []
    for completion_id, answer in zip(completion_ids, answers):
        completion_text = TrainerTools().tokenizer.decode_to_text(completion_id.unsqueeze(0))
        completion_text = completion_text.replace('<pad>', '').strip()
        correct_answer = TrainerTools().tokenizer.decode_to_text(answer.unsqueeze(0))

        rewards.append(get_reward(completion_text, correct_answer))

    return rewards


if __name__ == '__main__':
    init_env()

    eval_prompts = [
        f'{reasoning_system_prompt}<user>朱莉正在读一本 120 页的书。昨天，她能读12页，今天，她读的页数是昨天的两倍。如果她明天想读剩下的一半页，她应该读多少页？</s><assistant>',
        f'{reasoning_system_prompt}<user>詹姆斯从事教学工作 40 年。他的搭档教书的时间比他少了10年。他们的综合经验有多长？</s><assistant>',
        f'{reasoning_system_prompt}<user>赫克托买了一盒口香糖。他给了托德 4 个，然后他给了艾丽莎的是托德的两倍，然后他给了鲍比 5 个，比他给艾丽莎的四倍还少。如果赫克托还剩下 6 个口香糖，那么赫克托总共购买了多少个口香糖？</s><assistant>',
        f'{reasoning_system_prompt}<user>如果艾琳每周工作 40 小时，她将赚取 500 美元，并且每加班一小时即可额外获得 20 美元。如果她上周工作了 50 小时，请计算她的总收入。</s><assistant>'
    ]

    trainer = GRPOTrainer(
        train_config=get_grpo_config(),
        reward_func=reward_func,
        eval_prompts=eval_prompts
    )

    trainer.train()
```

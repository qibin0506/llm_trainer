# llm_trainer

## 安装
``` python
pip3 install project_llm_trainer
```

## 数据集格式说明
可参考：[https://github.com/qibin0506/llm_trainer/blob/master/example/create_dataset.md](https://github.com/qibin0506/llm_trainer/blob/master/example/create_dataset.md)

## 训练参数说明
可参考：[train_configs.py](https://github.com/qibin0506/llm_trainer/blob/master/llm_trainer/train_configs.py)或者下面配置训练文件的说明


## 开启训练流程

### 配置环境变量
```python
def init_env():
    # 禁用并行策略
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    # 指定tokenizer目录
    os.environ['TOKEN_DIR'] = './tokens/'
    # 指定日志目录，里面包括（训练loss、lr监控、异常信息、断点续训信息等）
    os.environ['LOG_DIR'] = './log/'
    # 指定使用deepspeed训练时保存checkpoint的目录
    os.environ['DIST_CHECKPOINT_DIR'] = 'ckpt_dir'
    # 指定使用非deepspeed训练时保存checkpoint文件
    os.environ['CHECKPOINT_NAME'] = 'ckpt.pth'
    # 指定最多保存多少个checkpoint
    os.environ['CKPT_MAX_TO_KEEP'] = '2'
    # 是否保存最佳loss的checkpoint
    os.environ['SAVE_BEST_CHECKPOINT'] = '0' # or '1'
```

### 配置模型参数
参考 [https://github.com/qibin0506/llm-model](https://github.com/qibin0506/llm-model)

### 配置训练文件
```python
from llm_trainer import FileDataset

class PretrainDataset(FileDataset):
    def __init__(self):
        self.files = ['./pretrain0.pkl', './pretrain1.jsonl']

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx) -> str:
        return self.files[idx]
```

### 配置训练参数
```
def _get_train_config(
        n_epochs: int,
        real_batch_size: int,
        file_dataset: FileDataset,
        model_config: ModelConfig,
        train_stage: str
):
    last_checkpoint = './last_checkpoint.bin'
    if train_stage != 'pretrain':
        assert os.path.exists(last_checkpoint)

    init_state_dict = torch.load(last_checkpoint, weights_only=True) \
        if os.path.exists(last_checkpoint) else None

    gradient_accumulation_steps = 3
    eval_batch_interval = 10 if train_stage == 'grpo' or train_stage == 'ppo' else 100

    ds_config = train_configs.DsConfig(
        zero_config=train_configs.DsZero3Config(
            offload_param=train_configs.DsOffloadConfig() if train_stage == 'grpo' or train_stage == 'ppo' else None,
            offload_optimizer=train_configs.DsOffloadConfig() if train_stage == 'grpo' or train_stage == 'ppo' else None
        )
    )

    pretrain_config = train_configs.PretrainConfig(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kd_config=None
    ) if train_stage == 'pretrain' or train_stage == 'midtrain' else None

    sft_config = train_configs.SFTConfig(
        mask_prompt=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        kd_config=None
    ) if train_stage == 'cot' or train_stage == 'mix' else None

    ppo_config = train_configs.PPOConfig(
        ppo_epochs=1,
        ppo_batch_size=2,
        gradient_accumulation_steps=8,
        vf_coef=0.5,
        kl_beta=0.02,
        kl_estimator='k3',
        normalize_rewards=True,
        ref_model_checkpoint=init_state_dict,
        gen_max_new_tokens=2048,
        gen_temperature=1.0,
        gen_p=0.95,
    ) if train_stage == 'ppo' else None

    dpo_config = train_configs.DPOConfig(
        ref_model_checkpoint=init_state_dict,
        mask_prompt=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_beta=0.1,
        loss_label_smoothing=0.0,
        nll_loss_coef=0.2
    ) if train_stage == 'dpo' else None

    grpo_config = train_configs.GRPOConfig(
        grpo_steps=4,
        group_size=16,
        loss_beta=0.0,
        loss_clip_eps=3e-4,
        loss_clip_eps_high=4e-4,
        loss_importance_sampling_level='seq',
        gen_max_new_tokens=1024,
        gen_temperature=1.0,
        gen_k=None,
        gen_p=0.95,
        gen_suppress_tokens=None,
    ) if train_stage == 'grpo' else None

    min_lr_ratio = 0.1
    max_lr = -1
    warmup_iters = -1
    period = -1
    enable_lr_scheduler = False

    if train_stage == 'ppo':
        initial_lr = 5e-6
    elif train_stage == 'grpo':
        initial_lr = 1e-5
    elif train_stage == 'dpo':
        initial_lr = 1e-6
    elif train_stage == 'cot':
        enable_lr_scheduler = True
        initial_lr = 1e-5 * TrainerTools().parallel.world_size
        max_lr = 5e-5 * TrainerTools().parallel.world_size
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=80000,  # 82431
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    elif train_stage == 'mix':
        enable_lr_scheduler = True
        initial_lr = 1e-5 * TrainerTools().parallel.world_size
        max_lr = 5e-5 * TrainerTools().parallel.world_size
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=50000,  # 56498
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    elif train_stage == 'midtrain':
        enable_lr_scheduler = True
        initial_lr = 1e-4 * TrainerTools().parallel.world_size
        max_lr = 5e-4 * TrainerTools().parallel.world_size
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=1000000,  # 1059891
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    else:
        enable_lr_scheduler = True
        initial_lr = 1e-4 * TrainerTools().parallel.world_size
        max_lr = 5e-4 * TrainerTools().parallel.world_size
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs,
            all_data_size=11000000,  # 11533122
            batch_size=real_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    optim_config = train_configs.OptimConfig(
        enable_lr_scheduler=enable_lr_scheduler,
        initial_lr=initial_lr,
        warmup_iters=warmup_iters,
        max_lr=max_lr,
        min_lr=initial_lr * min_lr_ratio,
        cosine_annealing_period=period
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
        max_seq_len=model_config.max_position_embeddings,
        loss_config=train_configs.LossConfig(),
        optim_config=optim_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        init_state_dict=init_state_dict,
        eval_config=train_configs.EvalConfig(
            max_new_tokens=model_config.max_position_embeddings,
            eval_batch_interval=eval_batch_interval,
        ),
        pretrain_config=pretrain_config,
        sft_config=sft_config,
        ppo_config=ppo_config,
        dpo_config=dpo_config,
        grpo_config=grpo_config
    )

    return train_config


def get_pretrain_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=20,
        file_dataset=PretrainFileDataset(),
        model_config=get_model_config(long_context=False),
        train_stage='pretrain'
    )

def get_sft_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=4,
        file_dataset=COTFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='sft'
    )


def get_grpo_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=1,
        file_dataset=GRPOFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='grpo'
    )

def get_dpo_config():
    return _get_train_config(
        n_epochs=2,
        real_batch_size=2,
        file_dataset=DPOFileDataset(),
        model_config=get_model_config(long_context=True),
        train_stage='dpo'
    )

def get_ppo_config():
    return _get_train_config(
        n_epochs=1,
        real_batch_size=16,
        file_dataset=PPODataset(),
        model_config=get_model_config(long_context=True),
        train_stage='ppo'
    )
```

### 训练入口

预训练
``` python

from llm_trainer import Trainer
from utils import init_env, get_pretrain_stage0_config

if __name__ == '__main__':
    init_env()
    eval_prompts = ['测试prompt']

    trainer = Trainer(
        train_config=get_pretrain_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()
```

SFT
``` python
from llm_trainer import SFTTrainer
from utils import init_env, get_sft_config, get_eval_prompt


if __name__ == '__main__':
    init_env()

    eval_prompts = ['测试prompt']

    trainer = SFTTrainer(
        train_config=get_mix_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()
```

GRPO
``` python
from llm_trainer import GRPOTrainer
from utils import init_env, get_sft_config, get_eval_prompt

def reward_func(prompt_ids: torch.Tensor, completion_ids: torch.Tensor, answers: torch.Tensor) -> List[float]:
    rewards = []
    return rewards

if __name__ == '__main__':
    init_env()

    eval_prompts = ['测试prompt']

    trainer = GRPOTrainer(
        train_config=get_grpo_config(),
        reward_func=reward_func,
        eval_prompts=eval_prompts
    )

    trainer.train()
```

DPO
``` python
from llm_trainer import DPOTrainer
from utils import init_env, get_dpo_config, get_eval_prompt

if __name__ == '__main__':
    init_env()

    eval_prompts = ['测试prompt']

    trainer = DPOTrainer(
        train_config=get_dpo_config(),
        eval_prompts=eval_prompts
    )

    trainer.train()
```

PPO
``` python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from typing import List, Optional
import torch
from llm_trainer import PPOTrainer, TrainerTools
from utils import init_env, get_ppo_config, get_eval_prompt

init_env()


def reward_func(
        prompt_ids: List[torch.Tensor],
        completion_ids: torch.Tensor,
        answers: List[Optional[torch.Tensor]]) -> List[float]:
    scores = []
    return scores


if __name__ == '__main__':
    eval_prompts = ['测试prompt']

    trainer = PPOTrainer(
        train_config=get_ppo_config(),
        reward_func=reward_func,
        eval_prompts=eval_prompts
    )

    trainer.train()
```

### 开启训练
|  训练命令 | 解释 | 实例 |
| --- | --- | --- |
| smart_train | 会优先使用`deepspeed`，未安装`deepspeed`时会自动降级到`ddp`训练，支持deepspeed、ddp原生参数，需要在指定训练文件后设置 | smart_train train_pretrain.py [raw arguments] |
| ds_train | 使用`deepspeed`训练 | ds_train train_pretrain.py [raw arguments] |
| ddp_train | 使用`ddp`训练 | ddp_train train_pretrain.py [raw arguments] |

* 建议使用`smart_train train_pretrain.py`开启训练


## 推理
``` python
from llm_trainer import TrainerTools, streaming_generate
generator = streaming_generate(
                model=model,
                prompt=prompt_token,
                max_position_embeddings=2048,
                max_new_tokens=2048,
                temperature=temperature,
                k=None,
                p=top_p,
                device=device
            )

for chunk in generator:
    print(chunk)
```

## 其他功能

### TrainerTools使用
TrainerTools是一个单例类，可以在外部环境使用训练中的一些实例。

`TrainerTools().parallel`可以获取当前训练中正在使用的并行方式，例如在多卡训练时判断是否为主进程：`if TrainerTools().parallel.is_main_process: print(log)`。

`TrainerTools().tokenizer`可以获取当前训练中正在使用的tokenizer，可以参考下面Tokenizer使用方法。

### Tokenizer
``` python
# encode to token id
TrainerTools().tokenizer.encode('hello world') # return [0, 1, 2...]
TrainerTools().tokenizer.encode('hello world', unsqueeze=True) # return torch.tensor([[0, 1, 2...]])
TrainerTools().tokenizer.encode('hello world', covert_tensor=True) # return torch.tensor([0, 1, 2...])

# decode from token id
TrainerTools().tokenizer.decode(torch.tensor([1, 2, 3])) # return hello world

# apply chat template
template = [
        {'role': 'system', 'content': 'system'},
        {'role': 'user', 'content': 'user'},
        {'role': 'assistant', 'content': '<think>think</think><answer>answer</answer>'},
        {'role': 'assistant', 'think': 'think2', 'content': 'answer2'}
    ]

encoded_template = TrainerTools().tokenizer.apply_chat_template(template) # return [0, 1, 2, 3]
encoded_template = TrainerTools().tokenizer.apply_chat_template(template, add_answer_tag_for_assistant=True) # 会在content中添加<answer></answer>标签，如果原始数据中已经包含，可以不指定encoded_template = TrainerTools().tokenizer.apply_chat_template(template, unsqueeze=True) # return torch.tensor([[0, 1, 2, 3]])
TrainerTools().tokenizer.apply_chat_template(template, covert_tensor=True) # return torch.tensor([0, 1, 2, 3])
```

### 内置脚本
项目内置多个方便用户使用的脚本，除上面提到的`smart_train`、`ds_train`、`ddp_train`外，还有以下脚本可以使用
|  脚本 | 解释 | 实例 |
| --- | --- | --- |
| vis_log | 绘制训练日志曲线，包括loss、reward等 | vis_log ./log/log.txt |
| vis_lr | 绘制训练lr曲线，将学习率可视化 | vis_lr ./log/lr.txt |
| calc_intermediate_size | 根据hidden_size计算intermediate_size | calc_intermediate_size 1024 # 结果为2752 |

### 调整断点续训
本项目自动支持断点续训，大部分情况下无需手动干预，但是有时候也有干预的需求，例如：我训练到最后的时候崩溃了，这个时候我其实不想等待断点续训的，而是把前面训练的文件全部注释掉，使用固定的lr把最后这部分文件训练完成，这个时候可以通过修改steps.pt完成。

``` python
ckpt = torch.load('./log/steps.pt', weights_only=True)
ckpt['epoch'] = 0 # 重置训练epoch
ckpt['file_idx'] = 0 # 重置当前训练的文件index
ckpt['batch_idx'] = 0 # 重置当前训练的batch
ckpt['cur_lr'] = 0.0018589864724561254 # 指定当前lr
ckpt['lr_steps'] = 0 # 重置lr部分
ckpt['cosine_annealing_base_lr'] = 0.002 # 重置余弦退火基础lr
ckpt['t_cur'] = 0.002 # 重置余弦退火当前周期内已走过的步数
ckpt['cycle'] = 0.002 # 重置余弦退火周期编号

torch.save(ckpt, './log/steps.pt')
```


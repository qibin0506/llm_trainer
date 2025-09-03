# llm_trainer

## 安装
``` python
pip3 install project_llm_trainer
```

## 训练参数说明
|  字段 | 类型 | 解释 |
|  ---- |  ----   | ---- |
| n_epochs | int | 指定训练轮数 |
| batch_size | int | 指定batch size |
| model_config | Union[ModelConfig, VLMConfig] | 指定模型配置，参考[https://github.com/qibin0506/llm-model](https://github.com/qibin0506/llm-model) |
| file_dataset | FileDataset | 指定训练文件数据集 |
| data_loader_config.data_loader_pin_memory | bool | 指定dataloader是否pin memory |
| data_loader_config.data_loader_num_workers | int | 指定dataloader加载线程数 |
| data_loader_config.data_loader_shuffle | bool | 是否shuffle数据 |
| data_loader_config.data_loader_drop_last | bool | 最后一个batch不满足batch_size时，是否丢弃 |
| image_tags_file_dataset | Optional[FileDataset] | 训练VLM时，指定image tags训练文件 |
| loss_config.critical_tokens | Optional[List[int]] | 指定计算loss时需要加强的token |
| loss_config.critical_alpha | float | 指定计算loss时需要加强的token的alpha |
| loss_config.aux_loss_coef | Optional[float] | 训练MoE模型时，aux loss的系数 |
| optim_config.optim_type | str | 指定训练优化器类型，支持adam和lion |
| optim_config.enable_lr_scheduler | bool | 是否使用lr调度器 |
| optim_config.initial_lr | float | 初始化lr |
| optim_config.weight_decay | float | 优化器weight_decay |
| optim_config.warmup_iters | Optional[int] | warmup迭代轮次 |
| optim_config.max_lr | float | 最大lr |
| optim_config.min_lr | float | 最小lr |
| optim_config.cosine_annealing_period | Optional[int] | 余弦退火周期 |
| optim_config.cosine_annealing_period_mul | int | 余弦退火mul |
| optim_config.ds_config | DsConfig | deepspeed配置字段，支持配置zero_config、fp16_config、bf16_config、gradient_clipping、activation_checkpointing，字段跟deepspeed官网一直，参考：[https://www.deepspeed.ai/docs/config-json/](https://www.deepspeed.ai/docs/config-json/) |
| kd_config | KDConfig | 用于logits蒸馏时使用 |
| dpo_config.loss_beta | float | DPO训练loss beta值 |
| dpo_config.loss_label_smoothing | float | DPO训练loss loss_label_smoothing |
| dpo_config.loss_ipo | bool | 是否使用IPO loss |
| dpo_config.nll_loss_coef | float | DPO nll loss系数 |
| grpo_config.grpo_steps | int | GRPO训练时指定每次生成数据后迭代训练次数 |
| grpo_config.group_size | int | GRPO训练组大小 |
| grpo_config.mixup_alpha | float | GRPO训练同步ref_model参数mixup_alpha |
| grpo_config.loss_beta | float | GRPO loss beta |
| grpo_config.loss_clip_eps | float | GRPO loss clip ps low |
| grpo_config.loss_clip_eps_high | Optional[float] | GRPO loss clip eps high |
| grpo_config.loss_delta | Optional[float] | GRPO loss delta |
| grpo_config.loss_importance_sampling_level | str | 取值token或seq，当token时是GRPO，当seq时是GSPO |
| grpo_config.loss_type | str | grpo or bnpo or dr_grpo |
| grpo_config.gen_max_new_tokens | Optional[int] | GRPO训练生成数据时最大长度 |
| grpo_config.gen_temperature | Optional[float] | GRPO训练生成数据时temperature |
| grpo_config.gen_k | Optional[int] | GRPO训练生成数据时top k |
| grpo_config.gen_p | Optional[int] | GRPO训练生成数据时top p |
| grpo_config.gen_suppress_tokens | int | GRPO训练生成数据时抑制token id |
| mask_prompt | bool | 进行SFT时，计算loss是否mask掉prompt部分，默认为True |
| gradient_accumulation_steps | int | 梯度累积步数 |
| eval_batch_interval | int | 指定每隔多少轮，进行一次数据生成 |
| eval_config.max_new_tokens | int | eval时最大生成长度 |
| eval_config.temperature | float | eval时生成数据temperature |
| eval_config.top_p | float | eval时生成数据top p |
| eval_config.top_k | float | eval时生成数据top k |
| pixel_values_provider | Optional[Callable[[list[str]], torch.Tensor]] | 训练VLM时指定图片pixel提供者 |
| init_state_dict | Optional[Mapping[str, Any]] | 指定初始化checkpoint |
| freeze_llm_model | bool | 是否冻结llm参数，主要用于vlm训练 |


## 开启训练流程

### 配置环境变量
```python
def init_env():
    # 禁用并行策略
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    # 指定tokenizer类型
    os.environ['TOKENIZERS_TYPE'] = 'zh_llama'
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
    gradient_accumulation_steps = 3
    eval_batch_interval = 10 if train_stage == 'grpo' else 100

    ds_config = train_configs.DsConfig(
        zero_config=train_configs.DsZero3Config(
            offload_param=train_configs.DsOffloadConfig() if train_stage == 'grpo' else None,
            offload_optimizer=train_configs.DsOffloadConfig() if train_stage == 'grpo' else None
        )
    )

    dpo_config = train_configs.DPOConfig(
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
        gen_p=0.85,
        gen_suppress_tokens=None,
    ) if train_stage == 'grpo' else None

    lr_mul = TrainerTools().parallel.world_size
    min_lr_ratio = 0.1

    initial_lr = 1e-4
    max_lr = 5e-4
    warmup_iters = 2000
    period = 100_000_000

    optim_config = train_configs.OptimConfig(
        optim_type='adam',
        enable_lr_scheduler=True,
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

    init_state_dict = torch.load('./last_checkpoint.bin', weights_only=True) if os.path.exists('./last_checkpoint.bin') else None

    train_config = train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_batch_interval=eval_batch_interval,
        loss_config=train_configs.LossConfig(),
        dpo_config=dpo_config,
        grpo_config=grpo_config,
        optim_config=optim_config,
        ds_config=ds_config,
        data_loader_config=data_loader_config,
        kd_config=None,
        init_state_dict=init_state_dict,
        eval_config=train_configs.EvalConfig()
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

### 开启训练
|  训练命令 | 解释 | 实例 |
| --- | --- | --- |
| smart_train | 会优先使用`deepspeed`，未安装`deepspeed`时会自动降级到`ddp`训练 | smart_train train_pretrain.py |
| ds_train | 使用`deepspeed`训练 | ds_train train_pretrain.py |
| ddp_train | 使用`ddp`训练 | ddp_train train_pretrain.py |

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
| plot_loss | 绘制训练loss曲线，将loss可视化 | plot_loss ./log/log.txt |
| plot_lr | 绘制训练lr曲线，将学习率可视化 | plot_lr ./log/lr.txt |
| calc_intermediate_size | 根据hidden_size计算intermediate_size | calc_intermediate_size 1024 # 结果为2752 |

### 调整断点续训
本项目自动支持断点续训，大部分情况下无需手动干预，但是有时候也有干预的需求，例如：我训练到最后的时候崩溃了，这个时候我其实不想等待断点续训的，而是把前面训练的文件全部注释掉，使用固定的lr把最后这部分文件训练完成，这个时候可以通过修改steps.pt完成。

``` python
ckpt = torch.load('./log/steps.pt', weights_only=True)
ckpt['global_steps'] = 0 # 重置训练步数
ckpt['cur_lr'] = 0.0018589864724561254 # 指定当前lr
ckpt['lr_steps'] = 0 # 重置lr部分
ckpt['cosine_annealing_base_lr'] = 0.002 # 重置余弦退火基础lr
ckpt['t_cur'] = 0.002 # 重置余弦退火当前周期内已走过的步数
ckpt['cycle'] = 0.002 # 重置余弦退火周期编号

torch.save(ckpt, './log/steps.pt')
```


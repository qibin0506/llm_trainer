# llm\_trainer: 全流程大模型高效训练框架

`llm_trainer` 是一个轻量级但功能强大的大语言模型（LLM）及视觉语言模型（VLM）训练框架。它支持从**预训练 (Pretrain)**、**有监督微调 (SFT)** 到 **人类反馈强化学习 (RLHF)** 的全流程训练，并内置了对 DeepSpeed 的深度集成。

## ✨ 核心特性

*   **全生命周期支持**：覆盖 Pretrain、SFT、DPO (Direct Preference Optimization)、PPO (Proximal Policy Optimization) 以及 GRPO (Group Relative Policy Optimization)。
*   **多模态支持 (VLM)**：原生支持视觉语言模型训练，支持图片 Tag 处理与 Pixel Value 转换，可冻结 LLM 部分仅训练 Projector。
*   **高效数据加载**：支持 `.jsonl`、`.pkl` 以及 **`.npy` (Memory Mapped)** 格式，极大降低海量数据训练时的内存占用。
*   **灵活的并行策略**：内置 `smart_train` 脚本，自动识别环境并在 DeepSpeed (Zero 0/1/2/3)、DDP 和单机模式间切换。
*   **丰富的 Loss 实现**：内置 Critical Token Loss、Aux Loss、Knowledge Distillation (KD) Loss 以及多种 RL Loss 实现。
*   **实用工具箱**：包含 Tokenizer 封装、学习率可视化、Loss 曲线绘制、断点续训管理等工具。
*   **配套模型框架**：[https://github.com/qibin0506/llm_model](https://github.com/qibin0506/llm_model)。

## 🛠️ 安装

可以通过 pip 安装，或直接从源码安装：


``` Bash
# 直接安装
pip3 install project_llm_trainer

# 源码安装
git clone https://github.com/qibin0506/llm_trainer.git
cd llm_trainer
pip install -e .

```

## 🚀 快速开始

### 1. 配置环境变量

项目依赖环境变量来定位资源，请在运行前设置：

``` Python
import os

def init_env():
    # Tokenizer 路径 (必须)
    os.environ['TOKEN_DIR'] = './tokens/'
    # 日志与 Checkpoint 目录
    os.environ['LOG_DIR'] = './log/'
    # DeepSpeed Checkpoint 目录
    os.environ['DIST_CHECKPOINT_DIR'] = 'ckpt_dir'
    # 常用配置
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CKPT_MAX_TO_KEEP'] = '2' # 最多保留几个ckpt

```

### 2. 准备数据

数据格式支持灵活配置，推荐使用 `.npy` 格式以获得最佳性能。

*详细数据生成示例请参考 [example/create\_dataset.md](https://github.com/qibin0506/llm_trainer/blob/master/example/create_dataset.md)*。

### 3. 开启训练

#### 预训练 (Pretrain)

``` Python
from llm_trainer import Trainer
from utils import init_env, get_pretrain_config

if __name__ == '__main__':
    init_env()
    trainer = Trainer(
        train_config=get_pretrain_config(),
        eval_prompts=['测试prompt']
    )
    trainer.train()

```

#### 有监督微调 (SFT) & VLM

``` Python
from llm_trainer import SFTTrainer

# VLM 配置示例：可在 SFTConfig 中指定 pixel_values_provider
trainer = SFTTrainer(
    train_config=get_sft_config(), 
    eval_prompts=['<image>描述这张图片'],
    eval_image_tags=['./test.jpg'] # 如果是VLM
)
trainer.train()

```

#### 强化学习 (GRPO / PPO / DPO)

以 GRPO 为例：

``` Python
from llm_trainer import GRPOTrainer

# 自定义 Reward Function
def reward_func(prompts, completions, answers):
    return [1.0 if len(c) > 10 else 0.0 for c in completions]

trainer = GRPOTrainer(
    train_config=get_grpo_config(),
    reward_func=reward_func,
    eval_prompts=['测试一下']
)
trainer.train()

```

## 💻 配置调用代码参考 (Configuration Code)

以下代码展示了如何根据不同训练阶段（Pretrain, SFT, PPO 等）组装 `TrainConfig`。你可以参考此模板在项目中实现自己的配置逻辑。

``` python
from llm_trainer import train_configs, TrainerTools
from llm_model import ModelConfig
import torch
import math

def _get_train_config(
        n_epochs: int,
        real_batch_size: int,
        file_dataset,
        model_config: ModelConfig,
        train_stage: str
):
    # 1. 加载断点或参考模型权重
    init_state_dict = torch.load('./last_checkpoint.bin', weights_only=True) if os.path.exists('./last_checkpoint.bin') else None
    ref_checkpoint = torch.load('./sft.bin', weights_only=True) if os.path.exists('./sft.bin') else None
    
    # 2. 基础配置
    gradient_accumulation_steps = 3
    eval_batch_interval = 10 if train_stage in ['grpo', 'ppo'] else 100

    # 3. 学习率与 Scheduler 配置
    enable_lr_scheduler = True
    min_lr_ratio = 0.1
    warmup_iters = -1
    period = -1
    
    if train_stage == 'ppo':
        initial_lr = 1e-5
    elif train_stage == 'sft':
        max_lr = 2e-5
        initial_lr = 1e-7
        # 自动计算 warmup 和 cosine 周期
        warmup_iters, period = calc_lr_schedular_args(
            epochs=n_epochs, all_data_size=86600, 
            batch_size=real_batch_size, gradient_accumulation_steps=gradient_accumulation_steps
        )
    # ... 其他阶段配置

    optim_config = train_configs.OptimConfig(
        enable_lr_scheduler=enable_lr_scheduler,
        initial_lr=initial_lr,
        warmup_iters=warmup_iters,
        max_lr=max_lr,
        min_lr=max_lr * min_lr_ratio if max_lr > 0 else initial_lr * min_lr_ratio,
        cosine_annealing_period=period
    )

    # 4. DeepSpeed 配置
    ds_config = train_configs.DsConfig(
        zero_config=train_configs.DsZero1Config() # 使用 Zero-1
    )

    # 5. 各阶段专属配置
    sft_config = train_configs.SFTConfig(
        mask_prompt=True,
        gradient_accumulation_steps=gradient_accumulation_steps
    ) if train_stage == 'sft' else None

    ppo_config = train_configs.PPOConfig(
        ppo_epochs=4,
        ppo_batch_size=5,
        gradient_accumulation_steps=10,
        vf_coef=0.5,
        kl_beta=0.01,
        ref_model_checkpoint=ref_checkpoint,
        gen_max_seq_len=2048,
        # PPO 独立的 Value Model 优化器配置
        value_optim_config=train_configs.OptimConfig(...)
    ) if train_stage == 'ppo' else None
    
    # ... DPO, GRPO 配置类似

    # 6. 返回最终 TrainConfig
    return train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        dataset_block_size=model_config.max_position_embeddings,
        optim_config=optim_config,
        ds_config=ds_config,
        sft_config=sft_config,
        ppo_config=ppo_config,
        # ... 其他 config
    )

```

***

## ⚙️ 训练参数详解

所有配置均通过 `llm_trainer.train_configs` 中的 Dataclass 定义。以下是详细参数说明。

### 1. TrainConfig (主配置)

`TrainConfig` 是训练的核心入口，控制全局参数。

| **参数名**              | **类型**             | **说明**                                                        |
| :------------------- | :----------------- | :------------------------------------------------------------ |
| `n_epochs`           | `int`              | 训练的总轮数 (Epochs)                                               |
| `batch_size`         | `int`              | 每个 GPU 的微批次大小 (Micro Batch Size)                              |
| `model_config`       | `ModelConfig`      | 模型结构配置 (Hidden size, Layers 等)                                |
| `file_dataset`       | `FileDataset`      | 训练数据集实例                                                       |
| `dataset_block_size` | `int`              | 训练序列的最大长度 (Seq Len)。若为 `None` 则取模型的 `max_position_embeddings` |
| `init_state_dict`    | `dict`             | (可选) 初始化的模型权重，用于断点续训或加载预训练权重                                  |
| `data_loader_config` | `DataLoaderConfig` | 数据加载器配置 (见下文)                                                 |
| `loss_config`        | `LossConfig`       | 损失函数配置 (见下文)                                                  |
| `optim_config`       | `OptimConfig`      | 优化器配置 (见下文)                                                   |
| `ds_config`          | `DsConfig`         | DeepSpeed 配置 (见下文)                                            |
| `eval_config`        | `EvalConfig`       | 评估生成配置 (见下文)                                                  |

### 2. OptimConfig (优化器配置)

控制学习率调度和优化器行为。

| **参数名**                   | **类型**  | **默认值**  | **说明**                         |
| :------------------------ | :------ | :------- | :----------------------------- |
| `optim_type`              | `str`   | `'adam'` | 优化器类型，支持 `'adam'` 或 `'lion'`   |
| `enable_lr_scheduler`     | `bool`  | `False`  | 是否启用学习率调度器                     |
| `initial_lr`              | `float` | -        | **初始学习率** (Warmup 结束后的最高学习率)   |
| `min_lr`                  | `float` | -        | 最小学习率 (余弦退火的终点)                |
| `max_lr`                  | `float` | -        | (可选) 最大学习率，通常与 `initial_lr` 相同 |
| `warmup_iters`            | `int`   | `None`   | 预热步数                           |
| `weight_decay`            | `float` | `None`   | 权重衰减系数                         |
| `betas`                   | `tuple` | `None`   | Adam 或 Lion 的 beta 参数          |
| `cosine_annealing_period` | `int`   | `None`   | 余弦退火周期步数                       |

### 3. DsConfig (DeepSpeed 配置)

控制分布式训练策略。

| **参数名**                    | **类型**                            | **说明**                                         |
| :------------------------- | :-------------------------------- | :--------------------------------------------- |
| `zero_config`              | `DsZeROConfig`                    | ZeRO 优化配置 (`DsZero0Config` \~ `DsZero3Config`) |
| `fp16_config`              | `DsFp16Config`                    | FP16 混合精度配置 (`enabled=True/False`)             |
| `bf16_config`              | `DsBf16Config`                    | BF16 混合精度配置 (`enabled=True/False`)             |
| `gradient_clipping`        | `float`                           | 梯度裁剪阈值 (默认 1.0)                                |
| `activation_checkpointing` | `DsActivationCheckpointingConfig` | 激活重计算 (梯度检查点) 配置，用于节省显存                        |

### 4. 阶段专属配置

根据不同的 `Trainer`，需要传入对应的专属配置对象。

#### SFTConfig (有监督微调)

| **参数名**                       | **类型**     | **默认值** | **说明**                               |
| :---------------------------- | :--------- | :------ | :----------------------------------- |
| `mask_prompt`                 | `bool`     | `True`  | 是否在计算 Loss 时屏蔽 Prompt 部分             |
| `freeze_llm_model`            | `bool`     | `False` | 是否冻结 LLM 参数 (用于 VLM 训练)              |
| `pixel_values_provider`       | `Callable` | `None`  | (VLM) 根据 Image Tag 获取图片 Tensor 的回调函数 |
| `gradient_accumulation_steps` | `int`      | `1`     | 梯度累积步数                               |

#### DPOConfig (偏好优化)

| **参数名**                | **类型**  | **默认值** | **说明**               |
| :--------------------- | :------ | :------ | :------------------- |
| `ref_model_checkpoint` | `dict`  | -       | 参考模型 (Ref Model) 的权重 |
| `loss_beta`            | `float` | -       | DPO 的 KL 惩罚系数 beta   |
| `loss_label_smoothing` | `float` | `0.0`   | 标签平滑系数               |
| `nll_loss_coef`        | `float` | `None`  | (可选) NLL Loss 的辅助系数  |

#### PPOConfig (强化学习)

| **参数名**                  | **类型**  | **说明**                            |
| :----------------------- | :------ | :-------------------------------- |
| `ppo_epochs`             | `int`   | 每次采集数据后，PPO 更新的轮数                 |
| `ppo_batch_size`         | `int`   | PPO 更新时的 mini-batch 大小            |
| `vf_coef`                | `float` | Value Function Loss 的系数 (通常 0.5)  |
| `kl_beta`                | `float` | KL 散度惩罚系数                         |
| `kl_estimator`           | `str`   | KL 估计器类型 (`'k1'` 或 `'k3'`)        |
| `normalize_rewards`      | `bool`  | 是否对 Reward 进行标准化 (RunningMeanStd) |
| `gen_max_seq_len`        | `int`   | 生成采样的最大长度                         |
| `gen_temperature`        | `float` | 采样温度                              |
| `ref_model_checkpoint`   | `dict`  | 参考模型权重                            |
| `value_model_checkpoint` | `dict`  | (可选) 独立的 Value Model 权重           |

#### GRPOConfig (组相对策略优化)

| **参数名**           | **类型**  | **说明**                              |
| :---------------- | :------ | :---------------------------------- |
| `group_size`      | `int`   | 每组采样的样本数量 (G)                       |
| `grpo_steps`      | `int`   | 每批数据更新的步数                           |
| `loss_beta`       | `float` | KL 惩罚项系数 (GRPO 中通常设为 0 或很小)         |
| `loss_type`       | `str`   | Loss 类型，支持 `'grpo'` (默认) 或 `'bnpo'` |
| `mixup_alpha`     | `float` | 训练模型与 Ref 模型参数混合系数 (默认 1.0，即不混合)    |
| `gen_max_seq_len` | `int`   | 生成最大长度                              |

### 5. 其他配置

*   **DataLoaderConfig**: `num_workers`, `pin_memory`, `shuffle` (是否打乱数据)。
*   **EvalConfig**: `eval_batch_interval` (每隔多少 Batch 评估一次), `max_seq_len` (评估生成长度)。
*   **LossConfig**: `aux_loss_coef` (MoE 负载均衡 Loss 系数), `critical_tokens` (关键 Token ID 列表, 用于加权 Loss)。
*   **KDConfig**: 知识蒸馏配置，需提供 `teacher_logits_provider`。

***

## 🖥️ 启动脚本

项目内置了智能启动命令，无需手动编写复杂的 `torchrun` 或 `deepspeed` 指令。

| **命令**            | **描述**                                                      | **示例**                            |
| :---------------- | :---------------------------------------------------------- | :-------------------------------- |
| **`smart_train`** | **推荐**。自动检测环境。优先使用 DeepSpeed，未安装则降级为 DDP，单卡则使用 Python 原生运行。 | `smart_train train_pretrain.py`   |
| **`ds_train`**    | 强制使用 DeepSpeed 启动。                                          | `ds_train train_sft.py --arg1 v1` |
| **`ddp_train`**   | 强制使用 DDP (torchrun) 启动。                                     | `ddp_train train_ppo.py`          |

## 📊 可视化与其他工具

项目在 `scripts` 目录下提供了一系列辅助脚本：

*   **`vis_log`**: 绘制训练日志曲线（Loss, Reward, Aux Loss 等）。

    ``` Bash
    vis_log ./log/log.txt

    ```
*   **`vis_lr`**: 可视化学习率变化曲线。

    ``` Bash
    vis_lr ./log/lr.txt

    ```
*   **`calc_intermediate_size`**: 辅助计算模型参数（如 FFN 的 intermediate size）。

    ``` Bash
    calc_intermediate_size 4096 # 输入 hidden_size

    ```

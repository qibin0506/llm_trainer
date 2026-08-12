# 🚀 LLM/VLM 全流程分布式训练与强化学习框架

一个基于 PyTorch 与 DeepSpeed 构建的高性能、通用大语言模型（LLM）与视觉语言模型（VLM）训练框架。支持从**预训练（Pretrain）**、**监督微调（SFT）**、**直接偏好优化（DPO/ORPO/SimPO）** 到 **强化学习（PPO & GRPO全系列算子）** 的完整生命周期。

---

## 📋 目录
1. [项目特性](#-项目特性)
2. [项目结构](#-项目结构)
3. [环境准备与环境变量配置](#-环境准备与环境变量配置)
4. [数据格式与 Tokenizer 规范](#-数据格式与-tokenizer-规范)
5. [多硬件与分布式训练配置](#-多硬件与分布式训练配置)
    * [硬件支持 (CUDA/NPU/MLU/MPS)](#硬件支持)
    * [DeepSpeed ZeRO-2 / ZeRO-3 & Offload 配置](#deepspeed-zero-2--zero-3--offload-配置)
6. [核心训练模块指南](#-核心训练模块指南)
    * [1. 预训练 (Pretrain Trainer)](#1-预训练-pretrain-trainer)
    * [2. 监督微调 (SFT Trainer - LLM & VLM)](#2-监督微调-sft-trainer---llm--vlm)
    * [3. 偏好对齐 (DPO / ORPO / SimPO Trainer)](#3-偏好对齐-dpo--orpo--simpo-trainer)
    * [4. 近端策略优化 (PPO Trainer)](#4-近端策略优化-ppo-trainer)
    * [5. 组相对策略优化 (GRPO Trainer & 前沿变体)](#5-组相对策略优化-grpo-trainer--前沿变体)
7. [自定义生成服务 (Generation Services)](#-自定义生成服务-generation-services)
8. [实用工具 (Tools & Utilities)](#-实用工具-tools--utilities)
9. [附录：全量参数配置详解](#-实用工具-tools--utilities)

---

## 🔥 项目特性

* **全流程算法支持**：覆盖 Pretrain、SFT、DPO/ORPO/SimPO、PPO 以及 DeepSeek-R1 核心的 GRPO 及其衍生算子（BNPO, Dr-GRPO, CISPO, DAPO, LUSPO, SAPO, VESPO）。
* **多模态 (VLM) 训练**：支持多模态投影层冻结/微调、图像虚拟 Token 扩展与动态 Pixel Features 注入。
* **异构硬件支持**：原生适配 **NVIDIA CUDA (NCCL)**、**华为升腾 NPU (HCCL)**、**寒武纪 MLU (CNCL)**、**Apple Silicon (MPS)** 及 **CPU/Gloo**。
* **DeepSpeed 深度集成**：灵活配置 ZeRO-1/2/3、ZeRO-Offload (CPU/NVMe)、ZeRO++ 梯度/权重量化及激活检查点（Activation Checkpointing）。
* **内存高效的数据载入**：支持 `.npy` (内存映射 mmap)、`.jsonl` 和 `.pkl` 格式，支持大体量数据集零内存暴涨加载。
* **解耦生成服务**：内置单卡集中生成、并行广播生成以及**多轮 RL 交互环境服务（Multi-Turn RL）**。
* **知识蒸馏 (KD) & 灾难性遗忘缓解**：SFT/Pretrain 支持基于 Logits 的 KD 损失；PPO/GRPO 支持 PTX 混合预训练损失。

---

## 📁 项目结构

```text
├── __init__.py             # 统一导出入口
├── base_trainer.py         # 训练器基类 (生命周期管理、梯度累积、Checkpoint、LR调度器)
├── trainer.py              # 预训练 Trainer
├── sft_trainer.py          # 监督微调 SFT Trainer (支持 LLM & VLM)
├── dpo_trainer.py          # 偏好对齐 DPO Trainer (支持 DPO, ORPO, SimPO)
├── ppo_trainer.py          # 强化学习 PPO Trainer (支持 Value Model, GAE, KL Penalty)
├── grpo_trainer.py         # 强化学习 GRPO Trainer (支持组内归一化及多种前沿 Loss)
├── train_configs.py        # 全局配置类 dataclasses (Optim, DsConfig, GenerateConfig 等)
├── parallel.py             # 分布式并行抽象层 (DsParallel, NoneParallel, 多后端适配)
├── generation_service.py   # 生成服务 (SyncCentral, Parallel, MultiTurnRL)
├── generate_utils.py       # 自回归生成底层算子 (KV Cache, 核采样, 惩罚项, Prefix Cache)
├── loss.py                 # Loss 算子库 (CrossEntropy, KD, DPO, PPO, GRPO全系列)
├── dataset.py              # Dataset 实现类 (Pretrain, SFT, DPO, RL)
├── tokenizer.py            # NanoTokenizer 封装与 Chat Template 应用
├── partition_utils.py      # ZeRO-3 权重 Gather、Unwrap 与跨 Rank 同步工具
├── checkpoint.py           # Checkpoint / Steps 序列化与恢复
├── ds_checkpoint.py        # DeepSpeed Checkpoint 管理
├── scheduler.py            # Warmup Cosine LR 调度器及复合调度器
├── tools.py                # 辅助工具 (权重格式转换, 步数计算, 数据量估算)
├── log.py                  # 日志记录器
└── utils.py                # 常用数学算子, Mask 算子, 硬件辅助算子
```

---

## 🛠 环境准备与环境变量配置

本框架通过系统环境变量快速切换硬件后端与训练模式：

| 环境变量 | 可选值 / 默认值 | 说明 |
| :--- | :--- | :--- |
| `TOKEN_DIR` | 必须指定 (如 `./tokenizer_path`) | 分词器目录路径 |
| `PARALLEL_TYPE` | `ds` / `none` (默认: `none`) | 分布式类型：DeepSpeed 或 单卡/无并行 |
| `USER_BACKEND` | `nccl` / `hccl` / `cncl` / `gloo` | 手动指定 PyTorch 分布式 Backend（默认自动识别） |
| `COMPUTE_DTYPE` | `auto` / `bf16` / `fp16` / `fp32` | 模型计算精度（默认 `auto`） |
| `CHECKPOINT_DIR` | 默认: `./checkpoints` | Checkpoint 检查点保存路径 |
| `LOG_DIR` | 默认: `./log` | 日志输出路径 |
| `CKPT_MAX_TO_KEEP` | 默认: `2` | 最多保留的 DeepSpeed global checkpoint 数量 |

**使用示例：**
```bash
export TOKEN_DIR="./my_tokenizer"
export PARALLEL_TYPE="ds"
export COMPUTE_DTYPE="bf16"
export CHECKPOINT_DIR="./output_ckpts"
```

---

## 📄 数据格式与 Tokenizer 规范

### 1. Tokenizer 与 Chat Template
`Tokenizer` 基于 `NanoTokenizer` 封装，约定了专用的 Special Tokens 与对话模版格式：

- **Special Tokens**：`</s>` (end), `<pad>`, `<unk>`, `<user>`, `<assistant>`, `<system>`, `<think>`, `</think>`, `<answer>`, `</answer>`, `<image>`
- **应用 Chat Template**：
  ```python
  from tokenizer import Tokenizer
  
  tokenizer = Tokenizer()
  conversations = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "1+1等于几？"},
      {"role": "assistant", "think": "思考：这是一个基础数学题", "content": "1+1等于2。"}
  ]
  token_ids = tokenizer.apply_chat_template(conversations)
  # 输出格式：<system>...</s><user>1+1等于几？</s><assistant><think>思考：...</think><answer>1+1等于2。</answer></s>
  ```

### 2. 数据集格式支持
框架支持三种文件后缀：`.jsonl`、`.pkl`、`.npy`（推荐 `.npy`，支持 `mmap` 零内存消耗加载）。

- **SFT 数据格式** (`.jsonl`)：
  ```json
  [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
  ```
- **DPO 数据格式** (`.jsonl`)：
  ```json
  {
    "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "好的回答"}],
    "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "差的回答"}]
  }
  ```
- **RL (PPO/GRPO) 数据格式** (`.jsonl`)：
  ```json
  {
    "prompt": [{"role": "user", "content": "求解方程 x^2 = 4"}],
    "answer": "x = 2 或 x = -2"
  }
  ```

---

## ⚡ 多硬件与分布式训练配置

### 硬件支持
框架自动根据设备类型选择最优后端：
- **NVIDIA GPU**: `nccl` 后端，启用 TF32 与 CUDA 优化。
- **华为升腾 NPU**: `hccl` 后端，自动处理 NPU 内存与计算。
- **寒武纪 MLU**: `cncl` 后端。
- **Apple Silicon**: `mps` 设备模式。

### DeepSpeed ZeRO-2 / ZeRO-3 & Offload 配置

通过 Python API 可以非常直观地构建 DeepSpeed 配置：

```python
from train_configs import (
    DsConfig, DsZero3Config, DsOffloadConfig, 
    DsBf16Config, DsActivationCheckpointingConfig
)

# 构建 ZeRO-3 CPU Offload 配置
ds_config = DsConfig(
    gradient_clipping=1.0,
    zero_config=DsZero3Config(
        stage3_prefetch_bucket_size='auto',
        stage3_param_persistence_threshold='auto',
        offload_optimizer=DsOffloadConfig(device='cpu', pin_memory=True),
        offload_param=DsOffloadConfig(device='cpu', pin_memory=True),
        zero_quantized_weights=False, # ZeRO++ 特性
    ),
    bf16_config=DsBf16Config(enabled=True),
    activation_checkpointing=DsActivationCheckpointingConfig(
        partition_activations=True,
        cpu_checkpointing=False
    )
)
```

---

## 🔍 核心训练模块指南

所有 Trainer 均继承自 `BaseTrainer`，内置了断点续训 (Resume)、混合精度 (AMP)、自动学习率调度、日志记录与定时 Evaluation 评估。

---

### 1. 预训练 (Pretrain Trainer)

预训练阶段针对文本序列使用自回归 Cross-Entropy Loss（可叠加 Teacher 模型进行知识蒸馏）。

```python
from trainer import Trainer
from train_configs import TrainConfig, PretrainConfig, OptimConfig

train_config = TrainConfig(
    n_epochs=1,
    batch_size=8,
    dataset_block_size=2048,
    file_dataset=["path/to/data1.npy", "path/to/data2.npy"],
    model_config=your_model_config,
    optim_config=OptimConfig(initial_lr=3e-4, optim_type='adam'),
    pretrain_config=PretrainConfig(gradient_accumulation_steps=4),
    ds_config=ds_config
)

trainer = Trainer(
    train_config=train_config,
    eval_prompts=["Today is a good day", "The future of AI is"]
)
trainer.train()
```

---

### 2. 监督微调 (SFT Trainer - LLM & VLM)

SFT 支持 Prompt Masking（只对 Assistant 回答计算 Loss），并原生支持多模态视觉语言模型 (VLM) 训练。

#### A. LLM 监督微调
```python
from sft_trainer import SFTTrainer
from train_configs import TrainConfig, SFTConfig, OptimConfig

train_config = TrainConfig(
    n_epochs=3,
    batch_size=4,
    dataset_block_size=1024,
    file_dataset=["path/to/sft_data.jsonl"],
    model_config=llm_model_config,
    optim_config=OptimConfig(initial_lr=2e-5),
    sft_config=SFTConfig(
        mask_prompt=True, # 开启 Prompt 掩码
        gradient_accumulation_steps=2
    )
)

trainer = SFTTrainer(
    train_config=train_config,
    eval_prompts=["<user>你好！</s><assistant>"]
)
trainer.train()
```

#### B. VLM 多模态微调
```python
from llm_model import VLMConfig

vlm_config = VLMConfig(...) # 包含 vision_config 与 tokens_per_image

def pixel_provider(image_tags):
    # 根据 image_tags 动态加载并返回 [B, C, H, W] Image Pixel Tensor
    return load_pixel_tensor_by_tags(image_tags)

train_config.sft_config = SFTConfig(
    mask_prompt=True,
    gradient_accumulation_steps=2,
    image_tags_file_dataset=["path/to/image_tags.csv"],
    pixel_values_provider=pixel_provider,
    freeze_llm_model=True # 可选择冻结 LLM 底座，仅微调 Projector
)
```

---

### 3. 偏好对齐 (DPO / ORPO / SimPO Trainer)

`DPOTrainer` 统一支持 **DPO**、**ORPO** 与 **SimPO** 算法。

```python
from dpo_trainer import DPOTrainer
from train_configs import TrainConfig, DPOConfig

# 1. 经典 DPO 配置
dpo_cfg = DPOConfig(
    loss_type='dpo',
    loss_beta=0.1,
    ref_model_weights_path="path/to/ref_model_weights",
    gradient_accumulation_steps=2
)

# 2. SimPO 配置 (无需 Reference Model，速度极快)
simpo_cfg = DPOConfig(
    loss_type='simpo',
    loss_beta=2.0,
    simpo_gamma=0.5,
    gradient_accumulation_steps=2
)

# 3. ORPO 配置 (结合 SFT NLL Loss)
orpo_cfg = DPOConfig(
    loss_type='orpo',
    loss_beta=0.1,
    nll_loss_coef=1.0, # 强烈推荐配置 NLL Loss
    gradient_accumulation_steps=2
)

train_config.dpo_config = dpo_cfg # 替换为你需要的偏好算法配置

trainer = DPOTrainer(
    train_config=train_config,
    eval_prompts=["<user>请写一首关于秋天的诗</s><assistant>"]
)
trainer.train()
```

---

### 4. 近端策略优化 (PPO Trainer)

PPO 包含了 Actor (Policy) 模型与 Critic (Value) 模型，采用 GAE 优势估计，内置 Running Mean/Std 归一化与 KL 散度惩罚。

```python
from ppo_trainer import PPOTrainer
from train_configs import TrainConfig, PPOConfig, GenerateConfig, OptimConfig

def reward_function(prompt_ids, completion_ids, gt_answer_ids):
    # 返回 List[float]，表示每个生成 Response 的标量 Reward
    return [compute_score(c, g) for c, g in zip(completion_ids, gt_answer_ids)]

train_config.ppo_config = PPOConfig(
    ppo_epochs=1,
    ppo_batch_size=2,
    gradient_accumulation_steps=2,
    ref_model_weights_path="path/to/ref_model",
    value_model_weights_path="path/to/value_model",
    value_optim_config=OptimConfig(initial_lr=1e-5), # Critic 独立学习率
    kl_beta=0.02,
    clip_eps=0.2,
    normalize_rewards=True,
    generate_config=GenerateConfig(max_seq_len=512, temperature=0.7)
)

trainer = PPOTrainer(
    train_config=train_config,
    reward_func=reward_function,
    eval_prompts=["求解方程：2x + 5 = 11"]
)
trainer.train()
```

---

### 5. 组相对策略优化 (GRPO Trainer & 前沿变体)

GRPO (Group Relative Policy Optimization) 是 DeepSeek-R1 的核心强化学习算法，对同一个 Prompt 生成一个 Group 的采样结果，在组内计算相对 Advantage，**无需单独的 Value 模型**。

本框架支持多种 GRPO Loss 变体算子（配置在 `GRPOConfig.loss_type`）：
- `grpo`: 经典截断 GRPO 算子。
- `bnpo`: Batch-level 归一化 GRPO。
- `dr_grpo`: 带有长度正则项的 GRPO。
- `cispo`: 剪裁重要性采样加权算法。
- `dapo`: 解耦剪裁范围算法。
- `luspo` / `sapo` / `vespo`: 支持长度偏置消除与软温度平滑。

```python
from grpo_trainer import GRPOTrainer
from train_configs import TrainConfig, GRPOConfig, GenerateConfig

def reward_function(prompt_ids, completion_ids, gt_answer_ids):
    # 针对 batch_size * group_size 条样本计算奖励
    scores = []
    for comp, gt in zip(completion_ids, gt_answer_ids):
        score = rule_based_math_checker(comp, gt)
        scores.append(score)
    return scores

# 可选：PTX 混合预训练，防止强化学习阶段遗忘通用能力
def ptx_builder(prompt_ids_list, gt_answer_ids_list):
    # 返回拼接好的 Prompt + Answer Tensor 列表
    return [torch.cat([p, a]) for p, a in zip(prompt_ids_list, gt_answer_ids_list)]

train_config.grpo_config = GRPOConfig(
    grpo_epochs=1,
    grpo_batch_size=2,
    group_size=8, # 每个 Prompt 采样 8 个回答进行组内竞争
    gradient_accumulation_steps=2,
    loss_type='grpo', # 可选: 'bnpo', 'dr_grpo', 'cispo', 'dapo', 'luspo', 'sapo', 'vespo'
    loss_beta=0.04,   # KL 散度约束强度
    ptx_coef=0.1,     # PTX 预训练 Loss 融合权重
    generate_config=GenerateConfig(max_seq_len=1024, temperature=0.9, top_p=0.95)
)

trainer = GRPOTrainer(
    train_config=train_config,
    reward_func=reward_function,
    ptx_builder=ptx_builder,
    eval_prompts=["证明勾股定理：a^2 + b^2 = c^2"]
)
trainer.train()
```

---

## 🤖 自定义生成服务 (Generation Services)

在 RL (PPO/GRPO) 阶段，如果采用 DeepSpeed ZeRO-3 参数切分，在前向采样生成时频繁 Gather 权重可能带来开销。框架提供了多种专门解耦的生成服务类（通过 `generation_service` 参数传入 Trainer）：

### 1. `SyncCentralGenerationService`
由 Rank 0 保持一份非切分的评估/采样模型，汇总所有 Rank 的 Prompts 并统一集中生成，再将结果 Broadcast 回各个 Rank。

### 2. `ParallelGenerationService`
多卡并行生成服务。使用自定义的桶式 `dist.broadcast` 高效同步模型最新权重到各卡独立生成设备，避免 pickle 序列化开销。

### 3. `MultiTurnRLGenerationService` (多轮环境交互/ Agent RL)
专门用于大模型代码执行、公式推理、工具调用的多轮强化学习交互服务。

```python
from generation_service import MultiTurnRLGenerationService

def environment_step(generated_text: str) -> tuple[bool, str]:
    # 提取代码并运行 Python 解释器
    code = extract_code(generated_text)
    success, output_or_error = python_interpreter.run(code)
    return success, output_or_error # (is_done, feedback)

def format_feedback(feedback_str: str) -> str:
    return f"\n<user>系统提示: 代码执行结果为:\n{feedback_str}</s>\n<assistant>"

gen_service = MultiTurnRLGenerationService(
    env_step=environment_step,
    format_feedback=format_feedback,
    max_turns=3,                  # 最多允许交互 3 轮
    max_consecutive_errors=2      # 连续报错 2 次自动终止该 Trajectory
)

# 传入 GRPOTrainer / PPOTrainer
trainer = GRPOTrainer(
    train_config=train_config,
    reward_func=reward_function,
    generation_service=gen_service,
    eval_prompts=eval_prompts
)
```

---

## 🛠 实用工具 (Tools & Utilities)

`tools.py` 包含了各种全流程开发所需的便捷辅助函数：

### 1. 自动计算学习率调度器总步数
根据训练阶段（SFT/DPO/PPO/GRPO）、数据量、Batch Size、卡数与 Warmup 比例，精确算得 `warmup_iters` 与 `cosine_annealing_batches`：

```python
from tools import compute_lr_scheduler_steps

warmup_iters, cosine_steps = compute_lr_scheduler_steps(
    train_stage='grpo',
    epochs=1,
    all_data_size=10000,
    batch_size=4,
    gradient_accumulation_steps=2,
    warmup_rate=0.03,
    grpo_epochs=1,
    group_size=8,
    grpo_batch_size=2
)
```

### 2. Checkpoint 权重导出为 Safetensors
将 DeepSpeed 导出的 FP32/ZeRO 检查点或者 PyTorch `.pt` 文件转存为标准的 `.safetensors` 格式：

```python
from tools import save_ds_weights_to_safetensors, save_pt_weights_to_safetensors

# DeepSpeed 检查点转换
save_ds_weights_to_safetensors(
    input_path="./checkpoints/global_step1000",
    output_path="./model.safetensors",
    dtype="bf16" # 可选 "fp16", "bf16", "fp32"
)

# PyTorch .pt 检查点转换
save_pt_weights_to_safetensors(
    input_path="./checkpoints/model.pth",
    output_path="./model.safetensors",
    dtype="bf16"
)
```

### 3. 从 PPO 复合权重中提取 Policy/Value 独立权重
```python
from tools import extract_policy_weights_from_ppo

# 提取纯 Policy 模型权重 state_dict，方便转存与部署
policy_state_dict = extract_policy_weights_from_ppo(model_config, ppo_checkpoint_weights)
```

---


# 📖 附录：`train_configs.py` 全量参数配置详解

本附录详细列出了框架中 `train_configs.py` 定义的所有 Dataclass 配置项、数据类型、默认值及其具体含义。

---

## 1. 全局训练主配置 (`TrainConfig`)

全局训练配置类，负责协调模型、优化器、分布式引擎及特定算法阶段的全局调度。

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `n_epochs` | `int` | **必填** | 全局数据集需要训练的总 Epoch 轮数。 |
| `batch_size` | `int` | **必填** | Micro-batch Size（每张 GPU 每次 Data Loader 取出的样本条数）。 |
| `model_config` | `Union[ModelConfig, VLMConfig]` | **必填** | 底层模型的架构配置定义（LLM 或 VLM 元配置）。 |
| `init_weights_path` | `Optional[str]` | `None` | 主干模型的初始化权重路径（本地目录或单文件）。 |
| `file_dataset` | `FileDataset` | **必填** | 训练数据集文件列表或 Dataset 接口实例。 |
| `dataset_block_size` | `int` | **必填** | 序列截断/打包的最大 Token 长度。 |
| `data_loader_config` | `DataLoaderConfig` | `DataLoaderConfig()` | PyTorch DataLoader 的加载配置（如 worker 进程数）。 |
| `optim_config` | `OptimConfig` | `OptimConfig()` | 优化器（Adam/Lion）与学习率调度器参数。 |
| `ds_config` | `DsConfig` | `DsConfig()` | DeepSpeed 分布式引擎配置（含 ZeRO、精度与检查点）。 |
| `eval_config` | `GenerateConfig` | `GenerateConfig()` | 训练过程中触发 Evaluation 阶段时的生成控制参数。 |
| `save_interval` | `int` | `100` | 每隔多少个 global batch step 触发一次保存 checkpoint。 |
| `eval_interval` | `int` | `100` | 每隔多少个 global batch step 触发一次测试集推理评估。 |
| `gradient_checkpointing` | `bool` | `False` | 是否开启梯度检查点（重计算）以节省显存。 |
| `pretrain_config` | `Optional[PretrainConfig]` | `None` | 使用 `Trainer` 进行无监督预训练时的特定配置。 |
| `sft_config` | `Optional[SFTConfig]` | `None` | 使用 `SFTTrainer` 进行监督微调时的特定配置。 |
| `dpo_config` | `Optional[DPOConfig]` | `None` | 使用 `DPOTrainer` 进行直接偏好对齐时的特定配置。 |
| `ppo_config` | `Optional[PPOConfig]` | `None` | 使用 `PPOTrainer` 进行 PPO 强化学习时的特定配置。 |
| `grpo_config` | `Optional[GRPOConfig]` | `None` | 使用 `GRPOTrainer` 进行 GRPO 强化学习时的特定配置。 |

---

## 2. 基础组件配置

### 2.1 优化器与学习率配置 (`OptimConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `optim_type` | `str` | `'adam'` | 优化器类型，可选 `'adam'` 或 `'lion'`。 |
| `auto_optimize_optimizer` | `bool` | `True` | 在 DeepSpeed 模式下是否自动替换为 Fused/CPU 优化器实现。 |
| `enable_lr_scheduler` | `bool` | `False` | 是否启用 Warmup Cosine 余弦退火学习率调度器。 |
| `initial_lr` | `float` | **必填** | 初始学习率（或经过 Warmup 后达到的峰值学习率）。 |
| `weight_decay` | `Optional[float]` | `None` | L2 正则化权重衰减系数（若为 `None`，Adam 默认为 0.01，Lion 默认为 0.015）。 |
| `betas` | `Optional[Tuple[float, float]]` | `None` | 优化器的 Beta 动量参数（若为 `None`，Adam 默认为 `(0.9, 0.999)`，Lion 为 `(0.95, 0.98)`）。 |
| `warmup_iters` | `Optional[int]` | `None` | 学习率线性预热的 Step 步数。 |
| `max_lr` | `Optional[float]` | `None` | 调度器允许的最大学习率。 |
| `min_lr` | `Optional[float]` | `None` | 余弦退火到达周期末尾时的最小下限学习率。 |
| `cosine_annealing_period` | `Optional[int]` | `None` | 余弦退火单周期包含的总 Step 步数。 |
| `cosine_annealing_period_mul` | `int` | `0` | 周期退火乘积系数；为 `0` 时表示不重复周期，超出后维持 `min_lr`。 |

### 2.2 数据加载配置 (`DataLoaderConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `pin_memory` | `bool` | `False` | 是否启用锁页内存（Pinned Memory）加速数据从 Host 传到 GPU。 |
| `num_workers` | `int` | `0` | PyTorch DataLoader 加载数据所使用的多进程子进程数。 |
| `shuffle` | `bool` | `False` | 是否在每个 Epoch 开始前随机打乱数据顺序。 |

### 2.3 生成解码配置 (`GenerateConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `max_seq_len` | `int` | `512` | 自回归生成的序列最大总长度（包含 Prompt）。 |
| `temperature` | `float` | `1.0` | 采样温度；值越大越随机，值为 `0` 或接近 `0` 时退化为贪婪搜索。 |
| `top_p` | `float` | `0.95` | Nucleus 采样概率阈值；仅保留累积概率达 `top_p` 的候选词。 |
| `top_k` | `Optional[int]` | `None` | Top-K 采样限制；设为非空时仅保留概率最高的 K 个候选词。 |
| `repetition_penalty` | `Optional[float]` | `1.0` | 重复惩罚因子；`> 1.0` 时惩罚已生成的重复 Token。 |
| `exclude_penalty_tokens` | `Optional[List[int]]` | `None` | 不受重复惩罚约束的豁免 Token ID 列表（如标点或特殊符号）。 |
| `suppress_tokens` | `Optional[List[int]]` | `None` | 强行抑制不被生成的 Token ID 列表（将 Logits 置为 `-inf`）。 |
| `auto_prefix_cache` | `bool` | `True` | 批量生成时是否开启公共 Prompt 前缀缓存优化。 |

### 2.4 知识蒸馏配置 (`KDConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `teacher_logits_provider` | `TeacherLogitsProvider` | **必填** | 外部提供的回调函数，用于获取 Teacher 模型针对当前 Batch 的软标签（Logits）。 |
| `kd_coef` | `float` | `0.4` | 蒸馏损失在总 Loss 中的融合权重：$\text{Total Loss} = \text{kd\_coef} \times \text{KD\_Loss} + (1 - \text{kd\_coef}) \times \text{Task\_Loss}$。 |

---

## 3. 各阶段训练算法配置

### 3.1 预训练配置 (`PretrainConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数，用于模拟更大 Global Batch Size。 |
| `kd_config` | `Optional[KDConfig]` | `None` | 知识蒸馏配置；设为 `None` 则不开启蒸馏。 |

### 3.2 监督微调配置 (`SFTConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `mask_prompt` | `bool` | `True` | 是否使用 `-100` Mask 屏蔽输入中的 Prompt，仅对回答部分计算 Loss。 |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数。 |
| `kd_config` | `Optional[KDConfig]` | `None` | 知识蒸馏配置。 |
| `image_tags_file_dataset` | `Optional[FileDataset]` | `None` | 多模态场景下提供图片 Tag 标识的文件映射 Dataset。 |
| `pixel_values_provider` | `Optional[PixelValuesProvider]` | `None` | 根据 Image Tag 动态提供图片像素 Tensor 的回调函数。 |
| `freeze_llm_model` | `bool` | `False` | VLM 微调时是否冻结 LLM 底座，仅训练 Projector 投影层。 |

### 3.3 偏好对齐配置 (`DPOConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ref_model_weights_path` | `Optional[str]` | `None` | 参考模型（Reference Model）的初始化权重路径。 |
| `mask_prompt` | `bool` | `True` | 是否屏蔽 Prompt 部分的 Loss。 |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数。 |
| `loss_type` | `str` | `'dpo'` | 偏好对齐损失算法类型，支持 `'dpo'`, `'orpo'`, `'simpo'`。 |
| `loss_beta` | `float` | **必填** | 偏好约束隐式 KL 惩罚强度参数（如 DPO 中通常设为 `0.1`）。 |
| `loss_label_smoothing` | `float` | `0.0` | DPO 的标签平滑系数（c-DPO 变体）。 |
| `loss_ipo` | `bool` | `False` | 是否切换为 IPO (Identity Preference Optimization) 的二次方 Loss 形式。 |
| `nll_loss_coef` | `Optional[float]` | `None` | 负对数似然 (NLL/SFT) 辅助损失权重系数（**ORPO 算法强推荐设为 1.0**）。 |
| `simpo_gamma` | `float` | `0.5` | SimPO 算法独有的 Target Reward Margin 间隔阈值。 |

### 3.4 近端策略优化配置 (`PPOConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ppo_epochs` | `int` | **必填** | 在当前 Batch 生成的 Rollout 数据上重复训练 Policy/Value 模型的轮数。 |
| `ppo_batch_size` | `int` | **必填** | PPO 内部计算前向/反向时的 Micro-batch 大小。 |
| `ref_model_weights_path` | `Optional[str]` | `None` | 参考模型的初始化权重路径。 |
| `value_model_weights_path` | `Optional[str]` | `None` | Critic (Value) 模型的初始化权重路径。 |
| `value_optim_config` | `Optional[OptimConfig]` | `None` | 为 Critic 模型配置的独立优化器与学习率设置。 |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数。 |
| `gamma` | `float` | `1.0` | GAE 优势估计中的折扣因子 $\gamma$。 |
| `lam` | `float` | `0.95` | GAE 优势估计中的平滑因子 $\lambda$。 |
| `clip_eps` | `float` | `0.1` | PPO 策略更新的代换截断阈值 $\epsilon$。 |
| `vf_coef` | `float` | `0.5` | 总 Loss 中 Value Loss 的比重系数。 |
| `kl_beta` | `float` | `0.02` | 基于 KL 散度的环境奖励惩罚系数。 |
| `kl_estimator` | `str` | `'k1'` | 近似 KL 计算公式：`'k1'`（Log-Ratio 方差）或 `'k3'`。 |
| `ptx_coef` | `float` | `0.0` | PTX 混合预训练 Loss 的权重，用于缓解灾难性遗忘。 |
| `missing_eos_penalty` | `Optional[float]` | `None` | 当生成的回答未能包含 EOS 结束符时的硬性扣分惩罚值。 |
| `normalize_rewards` | `bool` | `False` | 是否在送入 GAE 前对 Reward 进行标准化。 |
| `normalize_method` | `str` | `'RunningMeanStd'` | Reward 标准化算法：`'RunningMeanStd'` 或 `'BatchStd'`。 |
| `whiten_rewards` | `bool` | `False` | 是否对 GAE 计算得出的 Advantage 优势值进行白化（Whitening）处理。 |
| `generate_config` | `GenerateConfig` | `GenerateConfig()` | PPO 采样 Rollout 生成数据时的自回归解码参数。 |

### 3.5 组相对策略优化配置 (`GRPOConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `grpo_epochs` | `int` | **必填** | 同一批样本采样的迭代更新次数。 |
| `grpo_batch_size` | `int` | **必填** | 模型更新时的 Micro-batch 大小。 |
| `group_size` | `int` | `12` | 对同一个 Prompt 并行生成的不同的回答数量（用于组内归一化）。 |
| `ref_model_weights_path` | `Optional[str]` | `None` | 参考模型（Reference Model）权重路径；当 `loss_beta=0.0` 时可设为 `None`。 |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数。 |
| `loss_beta` | `float` | `0.04` | KL 散度惩罚系数。 |
| `loss_clip_eps` | `float` | `3e-4` | 组相对优化下限截断阈值 $\epsilon_{low}$。 |
| `loss_clip_eps_high` | `Optional[float]` | `4e-4` | 不对称截断时的上限阈值 $\epsilon_{high}$。 |
| `loss_delta` | `Optional[float]` | `None` | Advantage 权重的绝对值上限门限。 |
| `loss_importance_sampling_level` | `str` | `'token'` | 归一化重要性采样的作用层级：`'token'` 或 `'sequence'`。 |
| `loss_type` | `str` | `'grpo'` | GRPO 变体损失算子：`'grpo'`, `'bnpo'`, `'dr_grpo'`, `'cispo'`, `'dapo'`, `'luspo'`, `'sapo'`, `'vespo'`。 |
| `sapo_temperature_pos` | `float` | `1.0` | SAPO/VESPO 算法针对正优势样本的调节温度。 |
| `sapo_temperature_neg` | `float` | `1.0` | SAPO/VESPO 算法针对负优势样本的调节温度。 |
| `vespo_k_pos` | `float` | `2.0` | VESPO 算法特定 Gamma 权重超参数。 |
| `vespo_lambda_pos` | `float` | `3.0` | VESPO 算法特定 Gamma 权重超参数。 |
| `vespo_k_neg` | `float` | `3.0` | VESPO 算法特定 Gamma 权重超参数。 |
| `vespo_lambda_neg` | `float` | `2.0` | VESPO 算法特定 Gamma 权重超参数。 |
| `ptx_coef` | `float` | `0.0` | PTX 混合预训练 Loss 的权重系数。 |
| `generate_config` | `GenerateConfig` | `GenerateConfig()` | GRPO 组采样生成数据时的自回归解码参数。 |

---

## 4. DeepSpeed 分布式引擎配置

### 4.1 核心配置 (`DsConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `zero_config` | `Optional[DsZeROConfig]` | `DsZero3Config()` | ZeRO 阶段配置（可替换为 `DsZero0Config`, `DsZero1Config`, `DsZero2Config`, `DsZero3Config`）。 |
| `fp16_config` | `Optional[DsFp16Config]` | `DsFp16Config()` | FP16 混合精度控制选项。 |
| `bf16_config` | `Optional[DsBf16Config]` | `DsBf16Config()` | BF16 混合精度控制选项。 |
| `gradient_clipping` | `float` | `1.0` | 全局梯度裁剪阈值，防止梯度爆炸。 |
| `activation_checkpointing` | `Optional[DsActivationCheckpointingConfig]` | `None` | 激活检查点（显存重计算）高级选项。 |
| `wall_clock_breakdown` | `bool` | `False` | 是否输出前向/反向/通信耗时的柱状分析日志。 |
| `flops_profiler` | `Optional[DsFlopsProfilerConfig]` | `None` | 算子耗时与 TFLOPS 剖析器。 |

### 4.2 ZeRO 阶段配置项 (`DsZeROConfig` 及子类)

#### A. 基础属性 (`DsZeROConfig` 通用)
- `allgather_partitions` (`bool`, 默认 `True`): 是否自动 Gather 收集分布式的参数。
- `allgather_bucket_size` (`int`, 默认 `5e8`): All-Gather 通信桶字节大小。
- `overlap_comm` (`bool`, 默认 `True`): 是否开启通信与计算重叠（Overlap）。
- `reduce_scatter` (`bool`, 默认 `True`): 是否在 Backward 阶段使用 Reduce-Scatter 操作聚合梯度。
- `reduce_bucket_size` (`Union[str, int]`, 默认 `5e8`): Reduce-Scatter 通信桶大小。
- `contiguous_gradients` (`bool`, 默认 `True`): 是否在连续显存块中分配梯度。
- `ignore_unused_parameters` (`bool`, 默认 `False`): 是否忽略无梯度的参数（RLHF 旁路时建议设为 `True`）。
- `communication_data_type` (`Optional[str]`, 默认 `None`): 通信数据精度类型（如 `"fp16"` 或 `"bf16"`）。

#### B. ZeRO Stage 2 特有属性 (`DsZero2Config`)
- `offload_optimizer` (`Optional[DsOffloadConfig]`): 优化器状态 CPU/NVMe 卸载配置。
- `offload_param` (`Optional[DsOffloadConfig]`): 参数卸载配置。

#### C. ZeRO Stage 3 特有属性 (`DsZero3Config`)
- `sub_group_size` (`int`, 默认 `1e9`): 参数切分时的子通信组大小。
- `stage3_prefetch_bucket_size` (`Union[str, int]`, 默认 `'auto'`): 参数预取缓存桶大小。
- `stage3_param_persistence_threshold` (`Union[str, int]`, 默认 `'auto'`): 常驻显存不被切分的参数量阈值。
- `stage3_max_live_parameters` (`int`, 默认 `1e9`): 允许常驻在显存中的最大参数总量。
- `stage3_max_reuse_distance` (`int`, 默认 `1e9`): 决定参数是否保留备用的重用距离评估指标。
- `stage3_gather_16bit_weights_on_model_save` (`bool`, 默认 `True`): 保存 Checkpoint 时是否 Gather 收集半精度权重。
- `memory_efficient_linear` (`bool`, 默认 `True`): 是否在 Linear 算子中开启更精细的显存优化。
- `offload_optimizer` (`Optional[DsOffloadConfig]`): 优化器卸载配置。
- `offload_param` (`Optional[DsOffloadConfig]`): 权重参数 CPU/NVMe 卸载配置。
- `zero_quantized_weights` (`bool`, 默认 `False`): **ZeRO++ QWZ** 特性，开启 INT8 权重 All-Gather 传输，减少一半通信量。
- `zero_hpz_partition_size` (`int`, 默认 `1`): **ZeRO++ HPZ** 特性，层级切分策略。多机训练时建议设为单机 GPU 数（如 8），消除跨节点拉取权重的网络瓶颈。
- `zero_quantized_gradients` (`bool`, 默认 `False`): **ZeRO++ QGZ** 特性，开启 INT4/INT8 的梯度 Reduce-Scatter 压缩。

### 4.3 参数/优化器状态卸载配置 (`DsOffloadConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `device` | `str` | `'cpu'` | 卸载目标设备，可选 `'cpu'`（系统内存）或 `'nvme'`（高速 SSD）。 |
| `pin_memory` | `bool` | `True` | 是否使用锁页内存加速 CPU 到 GPU 的传输。 |
| `nvme_path` | `Optional[str]` | `None` | 当 `device='nvme'` 时，存放缓存张量的 NVMe 挂载目录路径。 |
| `buffer_count` | `Optional[int]` | `5` | NVMe 异步 I/O 缓冲区数量。 |
| `buffer_size` | `Optional[int]` | `100,000,000` | 单个 NVMe I/O 缓冲区大小（字节）。 |
| `max_in_cpu` | `Optional[int]` | `1,000,000,000` | NVMe 模式下允许在 CPU 内存中保留的最大数据上限（字节），防止 CPU OOM。 |

### 4.4 激活检查点配置 (`DsActivationCheckpointingConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `partition_activations` | `bool` | `True` | 是否跨 GPU 切分（Partition）激活张量以最大化节省显存。 |
| `cpu_checkpointing` | `bool` | `False` | 是否将激活检查点进一步卸载至系统 CPU 内存。 |
| `contiguous_memory_optimization` | `bool` | `True` | 是否开启连续内存分配优化，减少显存碎片化。 |
| `number_checkpoints` | `Optional[int]` | `None` | 手动指定的检查点数量。 |
| `synchronize_checkpoint_boundary` | `bool` | `False` | 是否在每个检查点边界强制执行 GPU 同步。 |
| `profile` | `bool` | `False` | 是否输出激活检查点的性能剖析日志。 |

### 4.5 性能与剖析配置 (`DsFlopsProfilerConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `False` | 是否开启 DeepSpeed Flops Profiler 性能分析器。 |
| `profile_step` | `int` | `1` | 指定在第几个 Global Step 开启性能剖析（通常跳过前几步以避开图构建）。 |
| `module_depth` | `int` | `-1` | 打印模型结构的深度层级；`-1` 表示无限制，打印全部细节。 |
| `top_modules` | `int` | `1` | 在性能分析报告中展示耗时或计算量排名前 N 的模块。 |
| `detailed` | `bool` | `True` | 是否打印包含各个算子及内存带宽详细信息的深度报告。 |
| `output_file` | `Optional[str]` | `None` | 分析报告写入的文件路径；若为 `None` 则直接打印到标准终端控制台。 |

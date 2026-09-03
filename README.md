# 🚀 LLM/VLM 全流程分布式训练与强化学习框架

一个基于 PyTorch 与 DeepSpeed 构建的高性能、通用大语言模型（LLM）与视觉语言模型（VLM）训练框架。支持从**预训练（Pretrain）**、**监督微调（SFT）**、**直接偏好优化（DPO/ORPO/SimPO）** 到 **强化学习（PPO & GRPO全系列算子）** 的完整生命周期。

---

## 📋 目录
1. [项目特性](#-项目特性)
2. [项目结构](#-项目结构)
3. [环境准备与环境变量配置](#-环境准备与环境变量配置)
4. [数据格式与 Tokenizer 规范](#-数据格式与-tokenizer-规范)
5. [优化器与调度器 (Muon / AdamW / Lion 混合优化)](#-优化器与调度器-muon--adamw--lion-混合优化)
    * [1. Muon 优化器与智能混合参数分组](#1-muon-优化器与智能混合参数分组)
    * [2. 多学习率协同调度 (Multi-Group LR Scaling)](#2-多学习率协同调度-multi-group-lr-scaling)
    * [3. DeepSpeed ZeRO-3 / Offload 与 Muon 深度协同](#3-deepspeed-zero-3--offload-与-muon-深度协同)
6. [多硬件与分布式训练配置](#-多硬件与分布式训练配置)
    * [硬件支持 (CUDA/NPU/MLU/MPS)](#硬件支持)
    * [DeepSpeed ZeRO-2 / ZeRO-3 & Offload 配置](#deepspeed-zero-2--zero-3--offload-配置)
7. [核心训练模块指南](#-核心训练模块指南)
    * [1. 预训练 (Pretrain Trainer & 分块交叉熵)](#1-预训练-pretrain-trainer)
    * [2. 监督微调 (SFT Trainer - ATF 与分块损失)](#2-监督微调-sft-trainer---llm--vlm)
    * [3. 偏好对齐 (DPO / ORPO / SimPO Trainer)](#3-偏好对齐-dpo--orpo--simpo-trainer)
    * [4. 近端策略优化 (PPO Trainer & Rollout分块)](#4-近端策略优化-ppo-trainer)
    * [5. 组相对策略优化 (GRPO Trainer & 前沿变体)](#5-组相对策略优化-grpo-trainer--前沿变体)
8. [自定义生成服务 (Generation Services)](#-自定义生成服务-generation-services)
9. [实用工具 (Tools & Utilities)](#-实用工具-tools--utilities)
10. [附录：全量参数配置详解](#-附录)

---

## 🔥 项目特性

* **全流程算法支持**：覆盖 Pretrain、SFT、DPO/ORPO/SimPO、PPO 以及 DeepSeek-R1 核心的 GRPO 及其衍生算子（BNPO, Dr-GRPO, CISPO, DAPO, LUSPO, SAPO, VESPO）。
* **新一代正交动量优化器 (Muon Optimizer) 与混合优化支持**：
  - **极分解正交更新**：原生实现 5 阶 Newton-Schulz 迭代（`MuonOptim`），支持 `torch.compile` 编译加速，显著提升大模型预训练与强化学习阶段的收敛速度与稳定性。
  - **智能混合参数分组 (Muon + AdamW)**：框架自动分离 2D 线性层矩阵（分配给 Muon 进行正交动量更新）与 1D 向量、Embedding、LM Head、LayerNorm/RMSNorm 及 Bias（分配给 AdamW/Lion），完全无需繁琐的手动配置。
  - **多学习率协同调度 (Multi-Group LR Scaling)**：内置比例学习率调度机制，Muon（如 `0.02`）与 AdamW（如 `1e-3`）即使基准学习率相差数十倍，其在 Warmup 和余弦退火阶段仍能保持严格按比例同步缩放。
  - **DeepSpeed ZeRO-1/2/3 深度适配**：无缝对接 DeepSpeed `MuonWithAuxAdam`，针对 ZeRO-3 与 CPU Offload 自动规避 `reduce_scatter` 限制，支持 `save_muon_momentum_buffer_in_memory`（常驻内存避免 NVMe 频繁换入换出）及 `zero_allow_untested_optimizer`。
  - **RL 全阶段支持**：PPO (Actor/Critic 独立或联合配置) 与 GRPO 均可一键开启 `optim_type='muon'`。
* **Active Token Filtering (ATF) & Chunked Cross Entropy (CCE)**：
  - **动态 Token 筛选**：在 SFT/Pretrain 阶段自动过滤 `-100` 掩码（Prompt/Padding），仅将有效 Token 送入投影层，砍掉 $70\% \sim 90\%$ 的无用 GEMM 计算。
  - **分块交叉熵 + 梯度检查点**：支持 `chunked_cross_entropy_size`，在前向阶段完全避免具象化超大 $[B, S, V]$ 的 Logits 显存。
  - **ZeRO-3 原生安全**：内置 `maybe_gather_lm_head_ctx` 与 Zero-Token 假节点穿透，彻底杜绝跨卡通信悬挂与死锁。
* **分块强化学习与自回归生成 (Chunked Generation & Log-probs)**：
  - **自回归生成分块 (`chunked_generate_size`)**：在 Rollout 采样阶段按 Chunk 执行 `batch_generate`，防止大 Batch/长文本并发 KV Cache 导致显存溢出（OOM）。
  - **评估前向分块 (`chunked_log_probs_size`)**：在 Policy 与 Reference 模型的 Log-probs 及 Value 评估阶段按 Chunk 分批推理，配合 Active Token Filtering 消除超长序列下的显存尖峰。
* **多模态 (VLM) 训练**：支持多模态投影层冻结/微调、图像虚拟 Token 扩展与动态 Pixel Features 注入。
* **异构硬件支持**：原生适配 **NVIDIA CUDA (NCCL)**、**华为升腾 NPU (HCCL)**、**寒武纪 MLU (CNCL)**、**Apple Silicon (MPS)** 及 **CPU/Gloo**。
* **DeepSpeed 深度集成**：灵活配置 ZeRO-1/2/3、ZeRO-Offload (CPU/NVMe)、ZeRO++ 梯度/权重量化及激活检查点（Activation Checkpointing）。
* **高效数据载入**：支持 `.npy` (内存映射 mmap)、`.jsonl` 和 `.pkl` 格式，大体量数据集零内存暴涨加载。
* **解耦生成服务**：内置单卡集中生成、并行广播生成以及**多轮 RL 交互环境服务（Multi-Turn RL）**。
* **分块知识蒸馏 (Chunked KD) & PTX**：支持在分块计算中同步完成 Student/Teacher 软标签蒸馏；PPO/GRPO 支持 PTX 混合预训练损失。

---

## 📁 项目结构

```text
├── __init__.py             # 统一导出入口
├── base_trainer.py         # 训练器基类 (生命周期管理、梯度累积、Checkpoint、优化器与参数分组管理)
├── trainer.py              # 预训练 Trainer (支持 ChunkedLMLoss 分块交叉熵)
├── sft_trainer.py          # 监督微调 SFT Trainer (支持 Active Token Filtering 与 VLM)
├── dpo_trainer.py          # 偏好对齐 DPO Trainer (支持 DPO, ORPO, SimPO)
├── ppo_trainer.py          # 强化学习 PPO Trainer (支持 Value Model, GAE, 生成与评估分块, Muon 混合优化)
├── grpo_trainer.py         # 强化学习 GRPO Trainer (支持组内归一化、生成与评估分块及多种前沿 Loss)
├── muon.py                 # Muon 优化器实现 (Newton-Schulz 正交迭代, MuonWithAdamW 及 DeepSpeed 集成)
├── train_configs.py        # 全局配置类 dataclasses (Optim, DsConfig, GenerateConfig, Protocols)
├── parallel.py             # 分布式并行抽象层 (DsParallel, NoneParallel, 多后端适配)
├── generation_service.py   # 生成服务 (SyncCentral, Parallel, MultiTurnRL)
├── generate_utils.py       # 自回归生成底层算子 (KV Cache, 核采样, 惩罚项, Prefix Cache)
├── loss.py                 # Loss 算子库 (ChunkedLMLoss, LMLoss, KDLoss, DPO, PPO, GRPO全系列)
├── dataset.py              # Dataset 实现类 (Pretrain, SFT, DPO, RL)
├── tokenizer.py            # NanoTokenizer 封装与 Chat Template 应用
├── partition_utils.py      # ZeRO-3 权重 Gather、maybe_gather_lm_head_ctx、Unwrap 与跨 Rank 同步
├── checkpoint.py           # Checkpoint / Steps 序列化与恢复
├── ds_checkpoint.py        # DeepSpeed Checkpoint 管理
├── scheduler.py            # Warmup Cosine LR 调度器 (支持多参数组比例缩放) 及复合调度器
├── tools.py                # 辅助工具 (权重格式转换, 步数计算, 数据量估算)
├── log.py                  # 日志记录器
└── utils.py                # 常用数学算子, Mask 算子, 硬件辅助算子, DeepSpeed 配置构建器
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

## ⚡ 优化器与调度器 (Muon / AdamW / Lion 混合优化)

框架支持 **AdamW**、**Lion** 以及前沿的 **Muon** 优化器，并提供了参数自动路由与跨组比例退火调度机制。

### 1. Muon 优化器与智能混合参数分组
Muon 对 2D 权重矩阵使用极分解正交化更新，但不能直接作用于 1D 偏置、LayerNorm、Embedding 或 LM Head。框架内置了**自动参数路由机制**：
- **2D 隐藏层投影矩阵**（如 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` 等）：自动路由至 `Muon` 组，使用 `muon_lr`、`muon_momentum` 及 5 阶 Newton-Schulz 正交迭代更新。
- **Embedding / LM Head / Norm / Bias / 1D 参数**：自动路由至 `AdamW` 衰减或非衰减组，使用 `initial_lr` 及 `betas` 更新。

```python
from train_configs import OptimConfig

# 配置 Muon 优化器 (自动混合 AdamW)
optim_config = OptimConfig(
    optim_type='muon',
    initial_lr=1e-3,                      # 适用于 AdamW 参数组（Embedding/Norm/Head/Bias）的学习率
    muon_lr=0.02,                         # 适用于 Muon 2D 矩阵权重专属学习率（建议 0.01 ~ 0.05）
    muon_momentum=0.95,                   # Muon 动量
    muon_ns_steps=5,                      # Newton-Schulz 迭代阶数
    muon_adjust_lr_fn='match_rms_adamw',  # 可选: 'original' 或 'match_rms_adamw' (按矩阵维度自适应调大学习率)
    weight_decay=0.01,
    enable_lr_scheduler=True,
    warmup_iters=200,
    cosine_annealing_period=5000
)
```

### 2. 多学习率协同调度 (Multi-Group LR Scaling)
当 Muon (`muon_lr=0.02`) 与 AdamW (`initial_lr=0.001`) 混合使用时，`WarmupCosineAnnealingLRScheduler` 会根据每个参数组各自的 `max_lr` 执行**比例衰减**（`lr_group = max_lr_group * (current_lr / max_lr)`），确保不同组在 Warmup、余弦退火及多周期重置时保持步调完全一致。

### 3. DeepSpeed ZeRO-3 / Offload 与 Muon 深度协同
- **自动兼容降级**：在 ZeRO-3 或 ZeRO-1/2 开启 CPU Offload 模式下，框架会自动将 `reduce_scatter` 置为 `False` 并开启 `zero_allow_untested_optimizer=True`，彻底解决 DeepSpeed 对自定义优化器的通信断言限制。
- **NVMe 极致性能优化**：在 ZeRO-3 启用 NVMe Offload 时，可通过 `DsZero3Config(save_muon_momentum_buffer_in_memory=True)` 将 Muon 动量缓冲区常驻系统内存，避免高频磁盘 I/O 带来的性能骤降。

---

## 🚀 多硬件与分布式训练配置

### 硬件支持
框架自动根据设备类型选择最优后端：
- **NVIDIA GPU**: `nccl` 后端，启用 TF32 与 CUDA 优化。
- **华为升腾 NPU**: `hccl` 后端，自动处理 NPU 内存与计算。
- **寒武纪 MLU**: `cncl` 后端。
- **Apple Silicon**: `mps` 设备模式。

### DeepSpeed ZeRO-2 / ZeRO-3 & Offload 配置

```python
from train_configs import (
    DsConfig, DsZero3Config, DsOffloadConfig, 
    DsBf16Config, DsActivationCheckpointingConfig
)

# 构建 ZeRO-3 CPU Offload 配置
ds_config = DsConfig(
    gradient_clipping=1.0,
    zero_allow_untested_optimizer=True, # 允许自定义优化器 (如 Muon)
    zero_config=DsZero3Config(
        stage3_prefetch_bucket_size='auto',
        stage3_param_persistence_threshold='auto',
        offload_optimizer=DsOffloadConfig(device='cpu', pin_memory=True),
        offload_param=DsOffloadConfig(device='cpu', pin_memory=True),
        zero_quantized_weights=False, # ZeRO++ 特性
        save_muon_momentum_buffer_in_memory=True # 使用 Muon 时的内存驻留优化
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

所有 Trainer 均继承自 `BaseTrainer`，内置断点续训 (Resume)、混合精度 (AMP)、Muon/AdamW 优化器智能配置、自动学习率调度、日志记录与定时 Evaluation 评估。

---

### 1. 预训练 (Pretrain Trainer)

预训练支持自回归 Cross-Entropy Loss（可叠加 Teacher 模型进行知识蒸馏）。通过配置 `chunked_cross_entropy_size` 可开启**分块交叉熵**，骨干模型自动跳过全量 Logits 计算（`return_logits=False`），大幅降低长文本预训练显存。

```python
from trainer import Trainer
from train_configs import TrainConfig, PretrainConfig, OptimConfig, KDConfig

train_config = TrainConfig(
    n_epochs=1,
    batch_size=8,
    dataset_block_size=4096,
    file_dataset=["path/to/data1.npy", "path/to/data2.npy"],
    model_config=your_model_config,
    optim_config=OptimConfig(
        optim_type='muon',              # 支持 'adam', 'lion', 'muon'
        initial_lr=1e-3,
        muon_lr=0.02
    ),
    pretrain_config=PretrainConfig(
        gradient_accumulation_steps=4,
        chunked_cross_entropy_size=1024 # 开启分块计算 (建议 512 ~ 2048)
    ),
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

SFT 原生支持 Prompt Masking（只对 Assistant 回答计算 Loss）。当配置 `chunked_cross_entropy_size` 时，底层通过 `ChunkedLMLoss` 自动激活 **Active Token Filtering (ATF)**：
- 自动将所有的有效 Token 进行紧凑排列并分块；
- 内部采用 `torch.utils.checkpoint` 逐块投影并释放显存；
- 支持在分块阶段同步完成 **Knowledge Distillation (KD)** 软标签蒸馏。

#### A. LLM 监督微调
```python
from sft_trainer import SFTTrainer
from train_configs import TrainConfig, SFTConfig, OptimConfig

train_config = TrainConfig(
    n_epochs=3,
    batch_size=4,
    dataset_block_size=2048,
    file_dataset=["path/to/sft_data.jsonl"],
    model_config=llm_model_config,
    optim_config=OptimConfig(initial_lr=2e-5, optim_type='adam'),
    sft_config=SFTConfig(
        mask_prompt=True,                 # 开启 Prompt 掩码
        gradient_accumulation_steps=2,
        chunked_cross_entropy_size=512    # 开启 Active Token Filtering 与分块 CE
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
    chunked_cross_entropy_size=512,
    image_tags_file_dataset=["path/to/image_tags.csv"],
    pixel_values_provider=pixel_provider,
    freeze_llm_model=True # 冻结 LLM 底座，仅微调 Projector
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

PPO 包含了 Actor (Policy) 模型与 Critic (Value) 模型，采用 GAE 优势估计，支持 Running Mean/Std 归一化、Advantage 白化（Whitening）与 KL 散度惩罚。
- 支持 Policy 与 Value 模型独立配置优化器类型（例如 Policy 使用 `muon`，Value 使用 `adam` 或 `muon`）。
- 支持配置 `GenerateConfig.chunked_generate_size` 在自回归采样生成阶段按分块执行，降低并发 KV Cache 显存峰值。
- 支持配置 `PPOConfig.chunked_log_probs_size` 在计算 Policy / Reference Log-probs 及 Value 评估阶段按分块执行，防止长序列显存溢出。

```python
from ppo_trainer import PPOTrainer
from train_configs import TrainConfig, PPOConfig, GenerateConfig, OptimConfig

# 支持 1D 轨迹标量奖励 或 2D 逐 Token 稠密奖励
def reward_function(prompt_ids, completion_ids, gt_answer_ids):
    # 返回 List[float] (1D Outcome Reward) 或 List[List[float]] (2D Process/Dense Reward)
    return [compute_score(c, g) for c, g in zip(completion_ids, gt_answer_ids)]

train_config.optim_config = OptimConfig(
    optim_type='muon', 
    initial_lr=1e-4, 
    muon_lr=0.02
)

train_config.ppo_config = PPOConfig(
    ppo_epochs=1,
    ppo_batch_size=2,
    gradient_accumulation_steps=2,
    chunked_log_probs_size=2,          # 估值与 Log-probs 评估阶段分批执行，消除峰值显存
    ref_model_weights_path="path/to/ref_model",
    value_model_weights_path="path/to/value_model",
    value_optim_config=OptimConfig(
        optim_type='adam', 
        initial_lr=1e-5                # Critic 独立优化器配置
    ), 
    kl_beta=0.02,
    clip_eps=0.1,
    vf_coef=0.1,
    normalize_rewards=False,
    generate_config=GenerateConfig(
        max_seq_len=512, 
        temperature=0.7,
        chunked_generate_size=2        # 自回归生成采样阶段分块大小
    )
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
from train_configs import TrainConfig, GRPOConfig, GenerateConfig, OptimConfig

def reward_function(prompt_ids, completion_ids, gt_answer_ids):
    # 针对 batch_size * group_size 条样本计算奖励 (支持 1D 标量或 2D 稠密打分)
    scores = []
    for comp, gt in zip(completion_ids, gt_answer_ids):
        score = rule_based_math_checker(comp, gt)
        scores.append(score)
    return scores

# 可选：PTX 混合预训练，防止强化学习阶段遗忘通用能力
def ptx_builder(prompt_ids_list, gt_answer_ids_list):
    # 返回拼接好的 Prompt + Answer Tensor 列表
    return [torch.cat([p, a]) for p, a in zip(prompt_ids_list, gt_answer_ids_list)]

train_config.optim_config = OptimConfig(
    optim_type='muon',
    initial_lr=5e-4,
    muon_lr=0.01
)

train_config.grpo_config = GRPOConfig(
    grpo_epochs=1,
    grpo_batch_size=2,
    group_size=12,                     # 组内采样数
    gradient_accumulation_steps=2,
    chunked_log_probs_size=4,          # Group 评估时分块计算 Log-probs，防止 OOM
    loss_type='grpo',                  # 可选: 'bnpo', 'dr_grpo', 'cispo', 'dapo', 'luspo', 'sapo', 'vespo'
    loss_beta=0.04,                    # KL 散度约束强度
    ptx_coef=0.1,                      # PTX 预训练 Loss 融合权重
    generate_config=GenerateConfig(
        max_seq_len=1024, 
        temperature=0.9, 
        top_p=0.95,
        chunked_generate_size=4        # 组内自回归采样分块大小
    )
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
由 Rank 0 保持一份非切分的评估/采样模型，汇总所有 Rank 的 Prompts 并统一集中生成（支持 `chunked_generate_size` 分块），再将结果 Broadcast 回各个 Rank。

### 2. `ParallelGenerationService`
多卡并行生成服务。使用自定义的桶式 `dist.broadcast` 高效同步模型最新权重到各卡独立生成设备，各卡本地按 `chunked_generate_size` 分块生成，避免 pickle 序列化开销。

### 3. `MultiTurnRLGenerationService` (多轮环境交互 / Agent RL)
专门用于大模型代码执行、公式推理、工具调用的多轮强化学习交互服务。在多轮自回归生成时原生支持分块生成与动态 Padding 对齐。

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

policy_state_dict = extract_policy_weights_from_ppo(model_config, ppo_checkpoint_weights)
```

---

# 📖 附录

## 1. 全局训练主配置 (`TrainConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `n_epochs` | `int` | **必填** | 全局数据集需要训练的总 Epoch 轮数。 |
| `batch_size` | `int` | **必填** | Micro-batch Size（每张 GPU 每次 Data Loader 取出的样本条数）。 |
| `model_config` | `Union[ModelConfig, VLMConfig]` | **必填** | 底层模型的架构配置定义（LLM 或 VLM 元配置）。 |
| `init_weights_path` | `Optional[str]` | `None` | 主干模型的初始化权重路径（本地目录或单文件）。 |
| `file_dataset` | `FileDataset` | **必填** | 训练数据集文件列表或 Dataset 接口实例。 |
| `dataset_block_size` | `int` | **必填** | 序列截断/打包的最大 Token 长度。 |
| `data_loader_config` | `DataLoaderConfig` | `DataLoaderConfig()` | PyTorch DataLoader 的加载配置（如 worker 进程数、shuffle 等）。 |
| `optim_config` | `OptimConfig` | `OptimConfig()` | 优化器（Adam/Lion/Muon）与学习率调度器参数。 |
| `ds_config` | `DsConfig` | `DsConfig()` | DeepSpeed 分布式引擎配置（含 ZeRO、精度与检查点）。 |
| `eval_config` | `GenerateConfig` | `GenerateConfig()` | 训练过程中触发 Evaluation 阶段时的生成控制参数。 |
| `save_interval` | `int` | `100` | 每隔多少个 global batch step 触发一次保存 checkpoint。 |
| `eval_interval` | `int` | `100` | 每隔多少个 global batch step 触发一次测试集推理评估。 |
| `gradient_checkpointing` | `bool` | `False` | 是否开启梯度检查点；若开启且使用 DeepSpeed，会自动同步初始化 `ds_config.activation_checkpointing`。 |
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
| `optim_type` | `str` | `'adam'` | 优化器类型，可选 `'adam'`, `'lion'`, `'muon'`。 |
| `auto_optimize_optimizer` | `bool` | `True` | 在 DeepSpeed 模式下是否自动替换为 Fused/CPU 优化器实现。 |
| `enable_lr_scheduler` | `bool` | `False` | 是否启用 Warmup Cosine 余弦退火学习率调度器。 |
| `initial_lr` | `float` | **必填** | 初始学习率（或经过 Warmup 后达到的峰值学习率；在使用 Muon 时作为 AdamW 组的基础学习率）。 |
| `weight_decay` | `Optional[float]` | `None` | L2 正则化权重衰减系数（若为 `None`，Adam 默认为 0.01，Lion 默认为 0.015）。 |
| `betas` | `Optional[Tuple[float, float]]` | `None` | 优化器的 Beta 动量参数（若为 `None`，Adam 默认为 `(0.9, 0.999)`，Lion 为 `(0.95, 0.98)`；Muon 的 AdamW 分支默认为 `(0.9, 0.95)`）。 |
| `warmup_iters` | `Optional[int]` | `None` | 学习率线性预热的 Step 步数。 |
| `max_lr` | `Optional[float]` | `None` | 调度器允许的最大学习率（默认与 `initial_lr` 相同）。 |
| `min_lr` | `Optional[float]` | `None` | 余弦退火到达周期末尾时的最小下限学习率（调度器内部默认为 `0.0`）。 |
| `cosine_annealing_period` | `Optional[int]` | `None` | 余弦退火单周期包含的总 Step 步数。 |
| `cosine_annealing_period_mul` | `int` | `0` | 周期退火乘积系数；为 `0` 时表示不重复周期，超出后维持 `min_lr`。 |
| `muon_lr` | `Optional[float]` | `0.02` | Muon 专用的初始学习率（通常建议 `0.01` ~ `0.05`），专门用于更新 2D 隐藏层权重矩阵。 |
| `muon_momentum` | `float` | `0.95` | Muon 的 Nesterov 动量因子。 |
| `muon_weight_decay` | `Optional[float]` | `None` | Muon 专属权重衰减系数；若为 `None` 则自动复用通用的 `weight_decay`（或 `0.01`）。 |
| `muon_ns_steps` | `int` | `5` | Muon 进行 5 阶 Newton-Schulz 极分解正交迭代的步数。 |
| `muon_adjust_lr_fn` | `Optional[str]` | `None` | 针对不同矩阵维度的学习率自适应调整函数，可选 `'original'`（按 $\sqrt{\max(1, A/B)}$ 缩放）或 `'match_rms_adamw'`（按 $0.2\sqrt{\max(A, B)}$ 匹配 AdamW RMS 尺度）。 |

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
| `chunked_generate_size` | `Optional[int]` | `None` | 自回归生成阶段 (`batch_generate`) 的 Batch 分块大小，防止并发 KV Cache 过大导致显存 OOM。 |
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
| `chunked_cross_entropy_size` | `Optional[int]` | `None` | 分块交叉熵大小（如 `512` / `1024` / `2048`）。开启后自动跳过全量 Logits 前向计算，按 Chunk 计算 CE/KD 损失以显著降低显存。 |

### 3.2 监督微调配置 (`SFTConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `mask_prompt` | `bool` | `True` | 是否使用 `-100` Mask 屏蔽输入中的 Prompt，仅对回答部分计算 Loss。 |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数。 |
| `kd_config` | `Optional[KDConfig]` | `None` | 知识蒸馏配置。 |
| `chunked_cross_entropy_size` | `Optional[int]` | `None` | 分块交叉熵大小。开启后自动启用 **Active Token Filtering (ATF)** 动态剔除 Prompt/Padding Token，并在 ZeRO-3 环境下安全分块计算损失。 |
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
| `value_optim_config` | `Optional[OptimConfig]` | `None` | 为 Critic 模型配置的独立优化器与学习率设置（支持独立配置 Muon 或 AdamW）。 |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数。 |
| `chunked_log_probs_size` | `Optional[int]` | `None` | 评估/前向阶段（计算旧策略及参考模型 Log Prob 和 Value）的 Batch 分块大小，用于降低显存峰值。 |
| `gamma` | `float` | `1.0` | GAE 优势估计中的折扣因子 $\gamma$。 |
| `lam` | `float` | `0.95` | GAE 优势估计中的平滑因子 $\lambda$。 |
| `clip_eps` | `float` | `0.1` | PPO 策略更新的代换截断阈值 $\epsilon$。 |
| `vf_coef` | `float` | `0.1` | 总 Loss 中 Value Loss 的比重系数。 |
| `kl_beta` | `float` | `0.02` | 基于 KL 散度的环境奖励惩罚系数。 |
| `kl_estimator` | `str` | `'k1'` | 近似 KL 计算公式：`'k1'`（Log-Ratio 方差）或 `'k3'`。 |
| `huber_delta` | `float` | `1.0` | Value 损失函数中 Smooth L1 (Huber Loss) 的平滑阈值 beta。 |
| `ptx_coef` | `float` | `0.0` | PTX 混合预训练 Loss 的权重，用于缓解灾难性遗忘。 |
| `missing_eos_penalty` | `Optional[float]` | `None` | 当生成的回答未能包含 EOS 结束符时的硬性扣分惩罚值。 |
| `normalize_rewards` | `bool` | `False` | 是否在送入 GAE 前对 Reward 进行标准化。 |
| `normalize_method` | `str` | `'RunningMeanStd'` | Reward 标准化算法：`'RunningMeanStd'` 或 `'BatchStd'`。 |
| `generate_config` | `GenerateConfig` | `GenerateConfig()` | PPO 采样 Rollout 生成数据时的自回归解码参数（可在此配置 `chunked_generate_size`）。 |

### 3.5 组相对策略优化配置 (`GRPOConfig`)

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `grpo_epochs` | `int` | **必填** | 同一批样本采样的迭代更新次数。 |
| `grpo_batch_size` | `int` | **必填** | 模型更新时的 Micro-batch 大小。 |
| `group_size` | `int` | `12` | 对同一个 Prompt 并行生成的不同的回答数量（用于组内归一化）。 |
| `ref_model_weights_path` | `Optional[str]` | `None` | 参考模型（Reference Model）权重路径；当 `loss_beta=0.0` 时可设为 `None`。 |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数。 |
| `chunked_log_probs_size` | `Optional[int]` | `None` | 评估/前向阶段（计算旧策略及参考模型 Log Prob）的 Batch 分块大小，用于降低显存峰值。 |
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
| `generate_config` | `GenerateConfig` | `GenerateConfig()` | GRPO 组采样生成数据时的自回归解码参数（可在此配置 `chunked_generate_size`）。 |

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
| `zero_allow_untested_optimizer` | `Optional[bool]` | `None` | 允许未在 DeepSpeed 官方白名单中测试的自定义优化器（ZeRO-1/2/3 通用；使用 `optim_type='muon'` 时若未指定，框架会自动开启为 `True`）。 |

### 4.2 ZeRO 阶段配置项 (`DsZeROConfig` 及子类)

#### A. 基础属性 (`DsZeROConfig` 通用)
- `allgather_partitions` (`bool`, 默认 `True`): 是否自动 Gather 收集分布式的参数。
- `allgather_bucket_size` (`int`, 默认 `5e8`): All-Gather 通信桶字节大小。
- `overlap_comm` (`bool`, 默认 `True`): 是否开启通信与计算重叠（Overlap）。
- `reduce_scatter` (`bool`, 默认 `True`): 是否在 Backward 阶段使用 Reduce-Scatter 操作聚合梯度（若使用 Muon 且在 ZeRO-3 / CPU Offload 模式下，框架会自动设为 `False` 以保证安全）。
- `reduce_bucket_size` (`Union[str, int]`, 默认 `5e8`): Reduce-Scatter 通信桶大小。
- `contiguous_gradients` (`bool`, 默认 `True`): 是否在连续显存块中分配梯度。
- `ignore_unused_parameters` (`bool`, 默认 `False`): 是否忽略无梯度的参数（RLHF 旁路时建议设为 `True`）。
- `communication_data_type` (`Optional[str]`, 默认 `None`): 通信数据精度类型（如 `"fp16"` 或 `"bf16"`）。

#### B. ZeRO Stage 2 特有属性 (`DsZero2Config`)
- `offload_optimizer` (`Optional[DsOffloadConfig]`): 优化器状态 CPU/NVMe 卸载配置。
- `offload_param` (`Optional[DsOffloadConfig]`): 注意：ZeRO-2 理论上只卸载优化器和梯度，参数卸载(offload_param)部分特性受限。

#### C. ZeRO Stage 3 特有属性 (`DsZero3Config`)
- `sub_group_size` (`int`, 默认 `1e9`): 参数切分时的子通信组大小。
- `stage3_prefetch_bucket_size` (`Union[str, int]`, 默认 `'auto'`): 参数预取缓存桶大小。
- `stage3_param_persistence_threshold` (`Union[str, int]`, 默认 `'auto'`): 常驻显存不被切分的参数量阈值。
- `stage3_max_live_parameters` (`int`, 默认 `1e9`): 允许常驻在显存中的最大参数总量。
- `stage3_max_reuse_distance` (`int`, 默认 `1e9`): 决定参数是否保留备用的重用距离评估指标。
- `stage3_gather_16bit_weights_on_model_save` (`bool`, 默认 `True`): 保存 Checkpoint 时是否 Gather 收集半精度权重。
- `memory_efficient_linear` (`bool`, 默认 `True`): 是否在 Linear 算子中开启更精细的显存优化。
- `offload_optimizer` (`Optional[DsOffloadConfig]`): 优化器卸载配置。
- `offload_param` (`Optional[DsOffloadConfig]`): 模型参数卸载配置。
- `zero_quantized_weights` (`bool`, 默认 `False`): **ZeRO++ QWZ** 特性，开启 INT8 权重 All-Gather 传输，减少一半通信量。
- `zero_hpz_partition_size` (`int`, 默认 `1`): **ZeRO++ HPZ** 特性，层级切分策略。多机训练时建议设为单机 GPU 数（如 8），消除跨节点拉取权重的网络瓶颈。
- `zero_quantized_gradients` (`bool`, 默认 `False`): **ZeRO++ QGZ** 特性，开启 INT4/INT8 的梯度 Reduce-Scatter 压缩。
- `save_muon_momentum_buffer_in_memory` (`Optional[bool]`, 默认 `None`): **ZeRO-3 + Muon 优化特性**。将 Muon 动量缓冲区常驻内存（配合 NVMe offload 极大提升吞吐并降低磁盘换入换出开销）。

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

---

## 5. Protocols 回调协议类型定义

框架在 `train_configs.py` 中定义了标准协议接口：

### 5.1 奖励计算接口 (`RewardFun`)
```python
class RewardFun(Protocol):
    def __call__(
        self,
        prompt_ids: List[torch.Tensor],
        completion_ids: torch.Tensor,
        gt_answer_ids: List[Optional[torch.Tensor]]
    ) -> Union[List[float], List[List[float]], torch.Tensor]:
        """
        支持返回：
        1. 1D 轨迹标量奖励 (List[float] 或 [N] Tensor): 结果导向，自动赋予序列最后一个有效 Token。
        2. 2D 逐 Token / 分步稠密奖励 (List[List[float]] 或 [N, max_completion_len] Tensor): 过程监督打分。
        """
        ...
```

### 5.2 自定义生成服务接口 (`GenerationService`)
```python
class GenerationService(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        prompt_ids: torch.Tensor,
        generate_config: GenerateConfig,
        task_type: str,
        pixel_values: Optional[torch.Tensor],
        tokens_per_image: Optional[int]
    ) -> Dict[str, Any]:
        """
        返回包含 'completions' (List[List[int]])、'dones' (Optional[List[bool]]) 
        及 'generation_masks' (Optional[List[List[bool]]]) 的字典。
        """
        ...
```

### 5.3 PTX 混合预训练构建器 (`PtxBuilder`)
```python
class PtxBuilder(Protocol):
    def __call__(
        self,
        prompt_ids: List[torch.Tensor],
        gt_answer_ids: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """返回长度为 [B] 的拼接后（Prompt + Answer）完整句子 Token 张量列表。"""
        ...
```

### 5.4 知识蒸馏与多模态接口 (`TeacherLogitsProvider` & `PixelValuesProvider`)
```python
class TeacherLogitsProvider(Protocol):
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: ...

class PixelValuesProvider(Protocol):
    def __call__(self, image_tags: List[str]) -> torch.Tensor: ...
```

# llm_trainer: 工业级大模型全流程高效训练框架

`llm_trainer` 是一个轻量级、高度解耦且功能强大的大语言模型（LLM）及视觉语言模型（VLM）训练框架。它不仅支持从**预训练 (Pretrain)**、**有监督微调 (SFT)** 到 **人类反馈强化学习 (RLHF)** 的全流程，更在底层深度集成了 DeepSpeed 的前沿特性，助你轻松驾驭LLM模型训练。

## ✨ 核心特性

*   **全生命周期对齐**：完整支持 Pretrain、SFT、DPO (直接偏好优化)、PPO (近端策略优化) 以及最新的 **GRPO (组相对策略优化)**。
*   **极致显存与通信优化**：深度集成 DeepSpeed ZeRO-1/2/3。原生支持 **ZeRO-Infinity (NVMe Offload)** 突破 CPU 内存限制，内置 **ZeRO++ (QWZ/HPZ/QGZ)** 极大压缩多机多卡通信带宽。
*   **原生多模态 (VLM) 支持**：支持图片 Tag 解析与 Pixel Value 动态映射，可一键冻结 LLM 底座仅微调 Projector 投影层。
*   **工业级 RLHF 容错机制**：内置 PPO/GRPO Reward 截断与白化、`ignore_unused_parameters` 防治计算图断裂、KL 散度动态约束等高阶稳定性保障。
*   **强类型极简配置**：全面采用 Python `Dataclass` 实现强类型配置树，提供极致的 IDE 代码补全提示，杜绝“配错参数导致 OOM”的低级错误。
*   **性能剖析与辅助工具箱**：内置 `FlopsProfiler` 分析真实 TFLOPS，配套智能启动脚本 `smart_train`、学习率可视化、Loss 曲线绘制工具。
*   **配套模型结构生态**：无缝对接底座 [https://github.com/qibin0506/llm_model](https://github.com/qibin0506/llm_model)。

## 🛠️ 安装

可以通过 pip 安装，或直接从源码安装：

```bash
# 直接安装
pip3 install project_llm_trainer

# 源码安装
git clone https://github.com/qibin0506/llm_trainer.git
cd llm_trainer
pip install -e .
```

## 🚀 快速开始

### 1. 配置环境变量

项目依赖环境变量来定位资源，请在运行主程序前设置：

```python
import os

def init_env():
    # Tokenizer 路径 (必须)
    os.environ['TOKEN_DIR'] = './tokens/'
    # 日志输出目录
    os.environ['LOG_DIR'] = './log/'
    # Checkpoint 存储目录
    os.environ['CHECKPOINT_DIR'] = './ckpt_dir/'
    # 最多保留的历史 Checkpoint 数量
    os.environ['CKPT_MAX_TO_KEEP'] = '2'
    # 常用 HuggingFace 环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### 2. 准备数据

数据加载器基于高度优化的读取逻辑，支持灵活配置，推荐使用 `.npy` (Memory Mapped) 格式以实现海量数据近乎零内存的极速加载。

*详细数据生成示例请参考 [example/create_dataset.md](https://github.com/qibin0506/llm_trainer/blob/master/example/create_dataset.md)*。

### 3. 开启训练

框架根据不同阶段提供不同的 `Trainer`，这里展示 5 种核心训练阶段的启动方式：

#### 预训练 (Pretrain)
```python
from llm_trainer import Trainer

trainer = Trainer(
    train_config=get_train_config(train_stage='pretrain'),
    eval_prompts=['续写这篇科幻小说的开头：']
)
trainer.train()
```

#### 有监督微调 (SFT) & 视觉大模型 (VLM)
```python
from llm_trainer import SFTTrainer

# VLM 训练：可在 SFTConfig 中指定 pixel_values_provider
trainer = SFTTrainer(
    train_config=get_train_config(train_stage='sft'), 
    eval_prompts=['<image>详细描述这张图片'],
    eval_image_tags=['./test.jpg'] # 为 Prompt 提供图像映射
)
trainer.train()
```

#### 直接偏好优化 (DPO)
```python
from llm_trainer import DPOTrainer

trainer = DPOTrainer(
    train_config=get_train_config(train_stage='dpo'),
    eval_prompts=['请解释量子力学']
)
trainer.train()
```

#### 近端策略优化 (PPO)
```python
from llm_trainer import PPOTrainer

# 自定义环境打分函数
def reward_func(prompts, completions, answers):
    return [1.0 if "def " in c else -1.0 for c in completions]

# (可选) 提供预训练数据生成器，混入 PTX Loss 缓解灾难性遗忘
def ptx_builder(prompts, answers):
    return [p + a for p, a in zip(prompts, answers)]

trainer = PPOTrainer(
    train_config=get_train_config(train_stage='ppo'),
    reward_func=reward_func,
    ptx_builder=ptx_builder,
    eval_prompts=['用 Python 写一个快速排序']
)
trainer.train()
```

#### 组相对策略优化 (GRPO)
```python
from llm_trainer import GRPOTrainer

# 自定义 Reward Function (按需返回规则/模型打分)
def reward_func(prompts, completions, answers):
    return [1.0 if len(c) > 10 else 0.0 for c in completions]

trainer = GRPOTrainer(
    train_config=get_train_config(train_stage='grpo'),
    reward_func=reward_func,
    eval_prompts=['请用 Python 写一个快排']
)
trainer.train()
```

---

## 💻 全量配置组装参考 (Configuration Template)

得益于精心设计的 `Dataclass` 配置层，你可以像搭积木一样安全、清晰地组装出适应任何模型的训练参数。**以下是一个完整且可直接复制套用的配置组装模板**：

```python
from llm_trainer import train_configs, TrainerTools
from llm_model import ModelConfig
import torch
import os

def get_train_config(
        n_epochs: int, 
        real_batch_size: int, 
        file_dataset, 
        model_config: ModelConfig, 
        train_stage: str
):
    # 1. 基础权重加载
    init_state_dict = torch.load('./last_checkpoint.bin', weights_only=True) if os.path.exists('./last_checkpoint.bin') else None
    
    # 强化学习阶段需要的参考模型 (Reference Model) 权重
    ref_checkpoint = torch.load('./sft_model.bin', weights_only=True) if os.path.exists('./sft_model.bin') else {"model_state_dict": {}}

    # 2. 生成阶段解码配置 (供 Eval 测试与 RL Rollout 生成交互使用)
    generate_config = train_configs.GenerateConfig(
        max_seq_len=2048,
        temperature=0.8,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.15,
        exclude_penalty_tokens=TrainerTools().tokenizer.encode('\n')
    )

    # 3. DeepSpeed 高阶优化引擎配置 (支持 ZeRO-3, NVMe Offload, ZeRO++)
    ds_config = train_configs.DsConfig(
        zero_config=train_configs.DsZero3Config(
            zero_quantized_weights=True,     # ZeRO++ 开启 INT8 权重通信压缩
            zero_hpz_partition_size=8,       # ZeRO++ 开启节点级层级切分 (多机设为单机卡数)
            offload_optimizer=train_configs.DsOffloadConfig(device='cpu') # 显存不足时可改为 'nvme'
        ),
        bf16_config=train_configs.DsBf16Config(enabled=True),
        flops_profiler=train_configs.DsFlopsProfilerConfig(enabled=False)
    )

    # 4. 学习率与主优化器配置
    optim_config = train_configs.OptimConfig(
        optim_type='adam',
        enable_lr_scheduler=True,
        initial_lr=1e-5 if train_stage in ['dpo', 'ppo', 'grpo'] else 2e-5,
        warmup_iters=100,
        min_lr=1e-6
    )

    # 5. 各个训练阶段专属配置初始化
    pretrain_config = None
    sft_config = None
    dpo_config = None
    ppo_config = None
    grpo_config = None

    if train_stage == 'pretrain':
        pretrain_config = train_configs.PretrainConfig(
            gradient_accumulation_steps=4
        )
        
    elif train_stage == 'sft':
        sft_config = train_configs.SFTConfig(
            mask_prompt=True,
            gradient_accumulation_steps=4
        )
        
    elif train_stage == 'dpo':
        dpo_config = train_configs.DPOConfig(
            ref_model_checkpoint=ref_checkpoint,
            loss_beta=0.1,
            gradient_accumulation_steps=4
        )
        
    elif train_stage == 'ppo':
        ppo_config = train_configs.PPOConfig(
            ppo_epochs=4,                 # 每批 Rollout 数据更新轮数
            ppo_batch_size=2,             # Micro-batch size
            gradient_accumulation_steps=4,
            ref_model_checkpoint=ref_checkpoint,
            vf_coef=0.5,
            kl_beta=0.02,
            ptx_coef=0.1,                 # 混入 10% 的 PTX Loss 防止遗忘
            generate_config=generate_config,
            # 独立的 Critic (Value) 模型优化器
            value_optim_config=train_configs.OptimConfig(
                optim_type='adam', initial_lr=2e-5, warmup_iters=100
            )
        )
        
    elif train_stage == 'grpo':
        grpo_config = train_configs.GRPOConfig(
            grpo_epochs=1,
            grpo_batch_size=2,
            group_size=8,                 # 每个 Prompt 生成 8 个答案计算相对优势
            gradient_accumulation_steps=4,
            loss_importance_sampling_level='token',
            generate_config=generate_config
        )

    # 6. 组装并返回最终全局 TrainConfig
    return train_configs.TrainConfig(
        n_epochs=n_epochs,
        batch_size=real_batch_size,
        model_config=model_config,
        file_dataset=file_dataset,
        dataset_block_size=model_config.max_position_embeddings,
        init_state_dict=init_state_dict,
        optim_config=optim_config,
        ds_config=ds_config,
        eval_config=generate_config,
        save_and_eval_interval=50 if train_stage in ['ppo', 'grpo'] else 200,
        
        # 挂载全部阶段配置 (框架会根据 Trainer 类型自动提取，None 会被安全忽略)
        pretrain_config=pretrain_config,
        sft_config=sft_config,
        dpo_config=dpo_config,
        ppo_config=ppo_config,
        grpo_config=grpo_config
    )
```

***

## ⚙️ 核心训练参数详解

项目所有的配置项均通过强类型的 `Dataclass` 进行定义（位于 `llm_trainer.train_configs`）。这种设计可以借助 IDE 提供完美的参数补全与类型检查。以下是系统所支持的所有配置参数详解。

### 1. TrainConfig (全局主配置)
控制整个训练周期的最高层级入口配置。

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `n_epochs` | `int` | 训练的总轮数 (Epochs)。 |
| `batch_size` | `int` | 每个 GPU 的数据加载微批次大小 (Micro Batch Size)。 |
| `model_config` | `ModelConfig` / `VLMConfig` | LLM 或 VLM 的模型结构定义与初始化参数。 |
| `init_state_dict` | `Mapping` | (可选) 初始化的模型权重，用于断点续训或预训练加载。 |
| `file_dataset` | `FileDataset` | 提供具体读取逻辑的数据集对象实例。 |
| `dataset_block_size` | `int` | 训练序列的截断长度。默认应与模型 `max_position_embeddings` 对齐。 |
| `data_loader_config` | `DataLoaderConfig` | DataLoader 加载器配置 (如 num_workers)。 |
| `optim_config` | `OptimConfig` | 全局主干优化器与学习率调度器配置。 |
| `ds_config` | `DsConfig` | DeepSpeed 引擎相关配置。 |
| `eval_config` | `GenerateConfig` | 触发边训边测 (Eval) 时模型的生成解码配置。 |
| `save_and_eval_interval` | `int` | 间隔多少个 global steps 自动保存 Checkpoint 并执行 Eval 测试。 |
| `pretrain_config` | `PretrainConfig` | **基础预训练阶段**专属的配置项包裹。 |
| `sft_config` | `SFTConfig` | **SFT (监督微调)** 专属的配置项包裹。 |
| `dpo_config` | `DPOConfig` | **DPO (直接偏好优化)** 专属的配置项包裹。 |
| `ppo_config` | `PPOConfig` | **PPO (近端策略优化)** 专属的配置项包裹。 |
| `grpo_config` | `GRPOConfig` | **GRPO (组相对策略优化)** 专属的配置项包裹。 |

---

### 2. DeepSpeed 引擎配置群 (DsConfig 及其子类)

#### DsConfig (DeepSpeed 顶层配置)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `zero_config` | `DsZeROConfig` | `DsZero3Config` | ZeRO 并行策略核心配置。支持 Stage 0~3。 |
| `fp16_config` | `DsFp16Config` | `DsFp16Config` | FP16 混合精度与动态 Loss Scale 配置。 |
| `bf16_config` | `DsBf16Config` | `DsBf16Config` | BF16 混合精度配置 (A100/H100 推荐开启)。 |
| `gradient_clipping` | `float` | `1.0` | 全局梯度裁剪阈值，防止梯度爆炸。 |
| `activation_checkpointing` | `DsActivation...` | `None` | 激活重计算配置，用计算时间换显存空间。 |
| `wall_clock_breakdown` | `bool` | `False` | 打印前向/反向/通信/更新的具体耗时比例分析。 |
| `flops_profiler` | `DsFlopsProfiler...` | `None` | DeepSpeed 算子性能与真实 TFLOPS 分析器。 |

#### DsZero3Config (ZeRO-3 与 ZeRO++ 核心参数)
*(注：DsZero0/1/2 享有下方部分属性)*

| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `stage` | `int` | `3` | ZeRO 并行阶段。 |
| `allgather_bucket_size` / `reduce_bucket_size` | `int` | `5e8` | 通信切分桶大小，控制网络数据包的规模。 |
| `overlap_comm` | `bool` | `True` | 开启计算与通信过程的重叠并行，掩盖网络延迟。 |
| `contiguous_gradients` | `bool` | `True` | 强制分配连续显存块来拷贝梯度，加速通信。 |
| `ignore_unused_parameters` | `bool` | `False` | **RLHF 必备！**设为 True 防止计算图中无梯度分支导致崩溃。 |
| `communication_data_type` | `str` | `None` | 指定梯度聚合通讯的精度 (`"fp16"`, `"bf16"`)，省带宽。 |
| `sub_group_size` | `int` | `1e9` | 节点内参数切分广播的组大小。 |
| `stage3_max_live_parameters` | `int` | `1e9` | 限制显存中同时存活的最大参数规模，防止 OOM。 |
| `stage3_gather_16bit_weights_on_model_save` | `bool` | `True` | 保存权重时是否自动聚合分布式状态。 |
| `memory_efficient_linear` | `bool` | `True` | 极度优化 ZeRO-3 状态下 Linear 算子的显存分配。 |
| `offload_optimizer` | `DsOffloadConfig` | `None` | 将优化器状态 (Momentum/Variance) 卸载。 |
| `offload_param` | `DsOffloadConfig` | `None` | 将模型权重本身从显存卸载。 |
| **`zero_quantized_weights`** | `bool` | `False` | **(ZeRO++ 特性)**：All-Gather 权重时采用 INT8 量化传输。 |
| **`zero_hpz_partition_size`** | `int` | `1` | **(ZeRO++ 特性)**：层级分片大小。多机训练设为单机 GPU 数（如 8）。 |
| **`zero_quantized_gradients`** | `bool` | `False` | **(ZeRO++ 特性)**：Reduce-Scatter 梯度时采用量化传输。 |

#### DsOffloadConfig (内存与 NVMe 卸载配置)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `device` | `str` | `'cpu'` | 卸载目标：系统内存 (`'cpu'`) 或 固态硬盘 (`'nvme'`)。 |
| `pin_memory` | `bool` | `True` | 是否使用锁页内存加速 CPU->GPU 数据拷贝。 |
| `nvme_path` | `str` | `None` | 如果 `device='nvme'`，存放缓存张量的磁盘路径。 |
| `buffer_count` / `buffer_size` | `int` | `5`, `1e8` | NVMe 异步读写的缓冲区控制参数。 |
| `max_in_cpu` | `int` | `1e9` | 控制最多允许多少字节留在系统内存，防止内存耗尽崩溃。 |

#### DsFlopsProfilerConfig (算子耗时与 FLOPs 诊断)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `enabled` | `bool` | `False` | 开启诊断工具。 |
| `profile_step` | `int` | `1` | 第多少个 Step 进行耗时剖析 (跳过前期 Warmup)。 |
| `module_depth` | `int` | `-1` | Profiler 打印模型嵌套层级的深度，-1 展开所有。 |
| `detailed` | `bool` | `True` | 打印详细的算子调用与带宽计算记录。 |

---

### 3. 各阶段专属训练配置

#### SFTConfig (有监督微调)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积，用时间换大 Batch Size。 |
| `mask_prompt` | `bool` | `True` | 将 Prompt 区间标签自动置为 -100，不计算 Loss。 |
| `freeze_llm_model` | `bool` | `False` | (用于 VLM) 冻结底座大模型，仅微调 Projector 网络。 |
| `pixel_values_provider` | `Callable` | `None` | (用于 VLM) 传入回调，通过图片路径/Tag 加载图像特征张量。 |
| `kd_config` | `KDConfig` | `None` | 知识蒸馏 (KD) 策略配置。 |

#### DPOConfig (直接偏好优化)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ref_model_checkpoint` | `Mapping` | 必须 | 参考模型权重。 |
| `loss_beta` | `float` | 必须 | DPO 算法的 KL 惩罚强度约束。 |
| `loss_label_smoothing` | `float` | `0.0` | c-DPO 标签平滑系数。 |
| `loss_ipo` | `bool` | `False` | 开启 IPO (Identity Preference Optimization) 损失形式。 |
| `nll_loss_coef` | `float` | `None` | DPO 的一种正规化手段：混入 NLL 损失，降低生成降级风险。 |

#### PPOConfig (近端策略优化)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `ppo_epochs` | `int` | 必须 | 一次 Rollout 的交互数据复用进行模型更新的迭代次数。 |
| `ppo_batch_size` | `int` | 必须 | 策略更新时的内循环 Mini-Batch Size。 |
| `ref_model_checkpoint` | `Mapping` | 必须 | 用于约束 Actor 不发生崩塌的基准参考模型。 |
| `value_model_checkpoint` | `Mapping` | `None` | 用于 Critic (Value Model) 的独立初始权重。 |
| `value_optim_config` | `OptimConfig`| `None` | Value 模型的独立优化器与学习率配置。 |
| `generate_config` | `GenerateConfig`| 默认 | 交互阶段 (Rollout) 的环境生成参数（如 max_len, temp）。 |
| `gamma` / `lam` | `float` | `1.0`, `0.95` | GAE 广义优势估计的折扣衰减与平滑权衡系数。 |
| `clip_eps` | `float` | `0.1` | PPO 核心的旧新策略分布限制截断比率。 |
| `vf_coef` / `kl_beta` | `float` | `0.5`, `0.02`| 价值网络损失系数，以及 KL 散度的基础惩罚项。 |
| `kl_estimator` | `str` | `'k1'` | 近似 KL 的估计器实现（支持 `k1`, `k3`）。 |
| `ptx_coef` | `float` | `0.0` | 在强化阶段混入一定比例的监督预训练 Loss 缓解遗忘。 |
| `normalize_rewards` | `bool` | `False` | 是否标准化环境返回的原始 Reward 标量。 |
| `normalize_method` | `str` | `'RunningMeanStd'`| 标准化方法选择 (全局流式滑动 vs 批次截取)。 |
| `whiten_rewards` | `bool` | `False` | 是否对 GAE 后计算出的优势 (Advantage) 进行白化截断。 |

#### GRPOConfig (组相对策略优化)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `group_size` | `int` | `12` | 对同一个 Prompt 并行生成多少个不同的答案 (用于 Advantage 标准化)。 |
| `grpo_epochs` | `int` | 必须 | 每批数据的复用更新迭代次数。 |
| `grpo_batch_size` | `int` | 必须 | Micro-Batch Size 大小。 |
| `loss_type` | `str` | `'grpo'` | 前沿算子切换：支持 `'grpo'`, `'bnpo'`, `'luspo'`, `'vespo'` 等。 |
| `loss_importance_sampling_level`| `str`| `'token'` | Token 级别 vs 序列序列整体级别的重采样截断。 |
| `mixup_alpha` | `float` | `1.0` | 采用 EMA 软更新将训练模型刷入 Ref 模型的动量。 |
| `loss_beta` | `float` | `0.04` | 组级 KL 惩罚强度。 |
| `sapo_temperature_pos`/`neg`| `float` | `1.0` | 针对前沿 SAPO 算法的正负梯度缩放温度。 |

---

### 4. 优化器、生成与辅助配置

#### OptimConfig (优化器与调度器)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `optim_type` | `str` | `'adam'` | 支持 `'adam'` (AdamW) 或 `'lion'`。 |
| `initial_lr` | `float` | 必须 | 基础学习率。 |
| `enable_lr_scheduler` | `bool` | `False` | 是否启动余弦退火学习率调度器。 |
| `warmup_iters` | `int` | `None` | 热身步数。 |
| `weight_decay` | `float` | `0.01` | L2 正则化权重。 |
| `max_lr` / `min_lr` | `float` | `None` | 调度波峰最大学习率，以及到达末尾或退火谷底的最小学习率。 |
| `cosine_annealing_period` | `int` | `None` | 退火周期。 |
| `cosine_annealing_period_mul` | `int` | `0` | 多周期衰减乘数。 |

#### GenerateConfig (文本生成解码配置)
| 参数名 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `max_seq_len` | `int` | `512` | 推理 / Rollout 截断的最大上下文长度。 |
| `temperature` | `float` | `1.0` | 采样温度，控制生成分布平滑度。 |
| `top_p` | `float` | `0.95` | Nucleus 截断阈值。 |
| `top_k` | `int` | `None` | 按最高概率 TopK 个词进行剪枝采样。 |
| `repetition_penalty` | `float` | `1.0` | 重复惩罚倍数，大于 1.0 时惩罚已有生成的词。 |
| `exclude_penalty_tokens` | `List` | `None` | 被豁免免受重复惩罚的 Token (如换行符、标点)。 |
| `suppress_tokens` | `List` | `None` | 将其 Logit 强制置为 -inf，绝对不让模型生成的 Token。 |

#### DataLoaderConfig & LossConfig & KDConfig
| 参数名 | 模块 | 说明 |
| :--- | :--- | :--- |
| `pin_memory` / `num_workers` / `shuffle` | `DataLoader` | PyTorch 数据集加载底层并发控制。 |
| `critical_tokens` / `critical_alpha` | `Loss` | 设置需重点监督的特殊符号 (如 `<eos>`, `</think>`) 及其损失加权倍数。 |
| `teacher_logits_provider` / `kd_coef`| `KDConfig` | 提供软标签 Teacher Logits 以及融合系数 (开启知识蒸馏)。 |

***


## 🖥️ 智能启动脚本

项目内置了针对单机与分布式的智能启动命令：

| **命令** | **描述** | **示例** |
| :--- | :--- | :--- |
| **`smart_train`** | **推荐**。自动检测当前环境，优先拉起 DeepSpeed 并行，若未安装则自动降级为 Python 原生运行。 | `smart_train train_sft.py` |
| **`ds_train`** | 强制使用 DeepSpeed 分布式引擎启动。 | `ds_train run_rlhf.py --arg1 v1` |

## 📊 可视化与诊断工具箱

在 `scripts` 目录下，附带了一系列提效脚本：

*   **`vis_log`**: 解析并可视化 `log.txt`，绘制 Loss、Reward 等多重指标曲线。
    ```bash
    vis_log ./log/log.txt
    ```
*   **`vis_lr`**: 绘制调度器产生的学习率 (Learning Rate) 预热与退火轨迹。
    ```bash
    vis_lr ./log/lr.txt
    ```
*   **`calc_intermediate_size`**: 辅助网络设计，计算大模型 FFN 层的规范化 intermediate size。
    ```bash
    calc_intermediate_size 4096  # 输入 Hidden Size
    ```

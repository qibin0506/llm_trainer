from typing import Optional, Union, Callable, List, Tuple
from dataclasses import dataclass, field

import torch
from llm_model import ModelConfig, VLMConfig
from .tools import FileDataset


@dataclass(kw_only=True)
class DsFlopsProfilerConfig:
    """
    DeepSpeed Flops Profiler 配置，用于分析模型前向/反向传播的算子耗时与真实 TFLOPS。

    Args:
        enabled (`bool`): 是否开启性能分析器。
        profile_step (`int`): 在指定的 global step 开启性能剖析（通常跳过前几步以绕过 warmup 和图图构建）。
        module_depth (`int`): 打印模型结构的深度层级，-1 表示没有限制，打印所有细节。
        top_modules (`int`): 在报告中展示耗时最长或计算量排名前 N 的子模块。
        detailed (`bool`): 是否打印包含各个算子及内存带宽详细信息的深度报告。
        output_file (`Optional[str]`): 报告写入的文件路径。若为 None，则直接打印到标准输出终端。
    """
    enabled: bool = False
    profile_step: int = 1
    module_depth: int = -1
    top_modules: int = 1
    detailed: bool = True
    output_file: Optional[str] = None


@dataclass(kw_only=True)
class DsOffloadConfig:
    """
    DeepSpeed 参数/优化器状态卸载 (Offload) 配置，支持 CPU 和 NVMe (ZeRO-Infinity) 卸载。

    Args:
        device (`str`): 卸载的目标设备，可选 'cpu' (系统内存) 或 'nvme' (高速固态硬盘)。
        pin_memory (`bool`): 是否使用锁页内存（Pinned Memory），这能显著加速 CPU 到 GPU 的数据传输。
        nvme_path (`Optional[str]`): 当 device='nvme' 时，指定用于存放缓存张量的 NVMe 挂载目录。
        buffer_count (`Optional[int]`): NVMe 异步 I/O 的缓冲区数量。
        buffer_size (`Optional[int]`): 单个 NVMe I/O 缓冲区的大小（字节）。
        max_in_cpu (`Optional[int]`): 限制在系统内存 (CPU RAM) 中保留的最大数据量（字节），防止 CPU OOM。
    """
    device: str = 'cpu'
    pin_memory: bool = True
    nvme_path: Optional[str] = None
    buffer_count: Optional[int] = 5
    buffer_size: Optional[int] = int(1e8)
    max_in_cpu: Optional[int] = int(1e9)


@dataclass(kw_only=True)
class DsActivationCheckpointingConfig:
    """
    DeepSpeed 激活检查点 (Activation Checkpointing/Gradient Checkpointing) 配置。

    Args:
        partition_activations (`bool`): 是否跨多张 GPU 切分（Partition）激活状态以节省极大的显存。
        cpu_checkpointing (`bool`): 是否将激活检查点进一步卸载到 CPU 内存中。
        contiguous_memory_optimization (`bool`): 是否启用连续内存优化，减少显存碎片化。
        number_checkpoints (`Optional[int]`): 检查点数量，用于在内存节省和重计算开销间寻找平衡。
        synchronize_checkpoint_boundary (`bool`): 是否在每个检查点边界强制进行 GPU 同步。
        profile (`bool`): 是否开启激活检查点的性能分析日志。
    """
    partition_activations: bool = True
    cpu_checkpointing: bool = False
    contiguous_memory_optimization: bool = True
    number_checkpoints: Optional[int] = None
    synchronize_checkpoint_boundary: bool = False
    profile: bool = False


@dataclass(kw_only=True)
class DsZeROConfig:
    """
    DeepSpeed ZeRO 基础配置基类。

    Args:
        stage (`int`): ZeRO 优化阶段，0 (无), 1 (优化器状态分区), 2 (梯度分区), 3 (参数分区)。
        allgather_partitions (`Optional[bool]`): 在每个 forward 结束时是否自动 Gather 收集分布式的参数。
        allgather_bucket_size (`Optional[int]`): All-Gather 通信的桶大小（字节），控制网络通信包的大小。
        overlap_comm (`Optional[bool]`): 是否重叠通信与计算过程，利用计算时间隐藏网络延迟。
        reduce_scatter (`Optional[bool]`): 是否在 backward 阶段使用 Reduce-Scatter 操作聚合梯度。
        reduce_bucket_size (`Optional[Union[str, int]]`): Reduce-Scatter 通信的桶大小（字节）。
        contiguous_gradients (`Optional[bool]`): 是否在连续的显存块中分配梯度，从而极大提升通信效率。
        ignore_unused_parameters (`Optional[bool]`): 是否忽略 forward 中未产生梯度的参数。RLHF 算法中存在某些旁路或特定步骤无梯度的情况，需设为 True 防止崩溃。
        communication_data_type (`Optional[str]`): 指定底层通信时使用的数据类型 (如 "fp16" 或 "bf16")，降低通信带宽高要求。
    """
    stage: int
    allgather_partitions: Optional[bool] = True
    allgather_bucket_size: Optional[int] = 5e8
    overlap_comm: Optional[bool] = True
    reduce_scatter: Optional[bool] = True
    reduce_bucket_size: Optional[Union[str, int]] = 5e8
    contiguous_gradients: Optional[bool] = True
    ignore_unused_parameters: Optional[bool] = False
    communication_data_type: Optional[str] = None  # "fp16" or "bf16"

@dataclass(kw_only=True)
class DsZero0Config(DsZeROConfig):
    """ZeRO Stage 0 配置 (等同于不进行 ZeRO 优化)。"""
    stage: int = field(default=0, init=False)


@dataclass(kw_only=True)
class DsZero1Config(DsZeROConfig):
    """ZeRO Stage 1 配置 (仅切割优化器状态)。"""
    stage: int = field(default=1, init=False)


@dataclass(kw_only=True)
class DsZero2Config(DsZeROConfig):
    """
    ZeRO Stage 2 配置 (切割优化器状态和梯度)。

    Args:
        offload_optimizer (`Optional[DsOffloadConfig]`): 优化器状态卸载配置。
        offload_param (`Optional[DsOffloadConfig]`): 注意：ZeRO-2 理论上只卸载优化器和梯度，参数卸载(offload_param)部分特性受限。
    """
    stage: int = field(default=2, init=False)
    offload_optimizer: Optional[DsOffloadConfig] = None
    offload_param: Optional[DsOffloadConfig] = None


@dataclass(kw_only=True)
class DsZero3Config(DsZeROConfig):
    """
    ZeRO Stage 3 (ZeRO-Infinity & ZeRO++) 配置 (切割优化器、梯度和参数)。

    Args:
        sub_group_size (`Optional[int]`): 参数切分时的子通信组大小。
        stage3_prefetch_bucket_size (`Optional[Union[str, int]]`): ZeRO-3 预取参数的缓冲区大小。
        stage3_param_persistence_threshold (`Optional[Union[str, int]]`): 控制多大的参数矩阵会常驻显存不被切分释放。
        stage3_max_live_parameters (`Optional[int]`): 允许常驻在显存中的最大参数量。
        stage3_max_reuse_distance (`Optional[int]`): 决定参数是否保留用于下次重用的评估距离指标。
        stage3_gather_16bit_weights_on_model_save (`Optional[bool]`): 在保存 Checkpoint 时是否自动 Gather 半精度权重。
        memory_efficient_linear (`Optional[bool]`): 在 Linear 算子中开启更精细的显存优化。
        offload_optimizer (`Optional[DsOffloadConfig]`): 优化器卸载配置。
        offload_param (`Optional[DsOffloadConfig]`): 模型参数卸载配置。
        zero_quantized_weights (`Optional[bool]`): ZeRO++ QWZ 特性，开启 INT8 权重 All-Gather 传输，减少一半通信量。
        zero_hpz_partition_size (`Optional[int]`): ZeRO++ HPZ 特性，层级切分策略。多机训练时建议设为单台机器的 GPU 数（如 8），消除机器间参数拉取的网络瓶颈。
        zero_quantized_gradients (`Optional[bool]`): ZeRO++ QGZ 特性，开启 INT4/INT8 的梯度 Reduce-Scatter，大幅压缩梯度通信量。
    """
    stage: int = field(default=3, init=False)
    sub_group_size: Optional[int] = 1e9
    stage3_prefetch_bucket_size: Optional[Union[str, int]] = 'auto'
    stage3_param_persistence_threshold: Optional[Union[str, int]] = 'auto'
    stage3_max_live_parameters: Optional[int] = 1e9
    stage3_max_reuse_distance: Optional[int] = 1e9
    stage3_gather_16bit_weights_on_model_save: Optional[bool] = True
    memory_efficient_linear: Optional[bool] = True
    offload_optimizer: Optional[DsOffloadConfig] = None
    offload_param: Optional[DsOffloadConfig] = None
    zero_quantized_weights: Optional[bool] = False
    zero_hpz_partition_size: Optional[int] = 1
    zero_quantized_gradients: Optional[bool] = False


@dataclass(kw_only=True)
class DsFp16Config:
    """
    DeepSpeed FP16 混合精度配置。

    Args:
        enabled (`Union[str, bool]`): 是否开启 FP16。
        loss_scale (`int`): 静态 loss scale 的值，若为 0 则使用动态缩放。
        loss_scale_window (`int`): 连续不发生 NaN/Inf 的 step 数量，达到后放大 scale。
        initial_scale_power (`int`): 动态 loss scale 初始化时的 2 的幂次方大小 (2^16)。
        hysteresis (`int`): 发生 NaN 后推迟放大 scale 的容忍周期。
        min_loss_scale (`int`): loss scale 可以下降到的最小下限。
        fp16_opt_level (`Optional[str]`): Apex 风格的优化级别（通常 O2）。
    """
    enabled: Union[str, bool] = 'auto'
    loss_scale: int = 0
    loss_scale_window: int = 1000
    initial_scale_power: int = 16
    hysteresis: int = 2
    min_loss_scale: int = 1
    fp16_opt_level: Optional[str] = 'O2'


@dataclass(kw_only=True)
class DsBf16Config:
    """
    DeepSpeed BF16 混合精度配置。BF16 的动态范围更大，通常无需 loss scale。

    Args:
        enabled (`bool`): 是否开启 BF16。
    """
    enabled: bool = True


@dataclass(kw_only=True)
class DsConfig:
    """
    DeepSpeed 引擎顶层封装配置。

    Args:
        zero_config (`Optional[DsZeROConfig]`): ZeRO 优化器核心配置 (阶段0/1/2/3)。
        fp16_config (`Optional[DsFp16Config]`): FP16 混合精度设置。
        bf16_config (`Optional[DsBf16Config]`): BF16 混合精度设置。
        gradient_clipping (`float`): 全局梯度裁剪阈值，防止梯度爆炸。
        activation_checkpointing (`Optional[DsActivationCheckpointingConfig]`): 激活重计算设置。
        wall_clock_breakdown (`bool`): 是否打印耗时拆解（Forward/Backward/Comm 等时间的占比分析）。
        flops_profiler (`Optional[DsFlopsProfilerConfig]`): DeepSpeed 算子性能与 TFLOPS 分析器配置。
    """
    zero_config: Optional[DsZeROConfig] = field(default_factory=DsZero3Config)
    fp16_config: Optional[DsFp16Config] = field(default_factory=DsFp16Config)
    bf16_config: Optional[DsBf16Config] = field(default_factory=DsBf16Config)
    gradient_clipping: float = 1.0
    activation_checkpointing: Optional[DsActivationCheckpointingConfig] = None
    wall_clock_breakdown: bool = False
    flops_profiler: Optional[DsFlopsProfilerConfig] = None


@dataclass(kw_only=True)
class DataLoaderConfig:
    """
    DataLoader 加载器配置。

    Args:
        pin_memory (`bool`): 是否使用锁页内存加速向 GPU 转移批次数据。
        num_workers (`int`): 数据加载使用的子进程数。
        shuffle (`bool`): 是否在每个 epoch 开始时随机打乱数据。
    """
    pin_memory: bool = False
    num_workers: int = 0
    shuffle: bool = False


@dataclass(kw_only=True)
class OptimConfig:
    """
    优化器及学习率调度器 (LR Scheduler) 核心配置。

    Args:
        optim_type (`str`): 优化器类型，支持 'adam', 'lion'。
        auto_optimize_optimizer (`bool`): 如果允许，是否由 DeepSpeed 自行替换并接管 CPU/Fused 优化器实现。
        enable_lr_scheduler (`bool`): 是否启用学习率调度器。
        initial_lr (`float`): 初始学习率 (或经过 warmup 后达到的最大学习率)。
        weight_decay (`Optional[float]`): 权重衰减系数 (L2 正则化)。
        betas (`Optional[Tuple[float, float]]`): Adam/Lion 的 beta 参数。
        warmup_iters (`Optional[int]`): 学习率线性预热的步数。
        max_lr (`Optional[float]`): 调度器允许的最大学习率。
        min_lr (`Optional[float]`): 余弦退火到达周期末尾时的最小学习率。
        cosine_annealing_period (`Optional[int]`): 余弦退火的一个完整周期的步数。
        cosine_annealing_period_mul (`int`): 周期的乘积系数，控制后续周期是否成倍变长。
    """
    optim_type: str = 'adam'
    auto_optimize_optimizer: bool = True
    enable_lr_scheduler: bool = False
    initial_lr: float
    weight_decay: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    warmup_iters: Optional[int] = None
    max_lr: Optional[float] = None
    min_lr: Optional[float] = None
    cosine_annealing_period: Optional[int] = None
    cosine_annealing_period_mul: int = 0


@dataclass(kw_only=True)
class KDConfig:
    """
    基于 Logits 级别的知识蒸馏 (Knowledge Distillation) 配置。

    Args:
        teacher_logits_provider (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            外部提供的回调函数，用于获取 Teacher 模型针对当前 Batch 生成的软标签 (Logits)。
        kd_coef (`float`, default=0.4):
            知识蒸馏 Loss 的融合占比。总 Loss = kd_coef * distil_loss + (1 - kd_coef) * task_loss。
    """
    teacher_logits_provider: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    kd_coef: float = 0.4


@dataclass(kw_only=True)
class GenerateConfig:
    """
    模型自回归生成 (Eval / Generation / Rollout) 阶段的解码配置。

    Args:
        max_seq_len (`int`): 生成序列的最大总长度（包含 Prompt）。
        temperature (`float`): 采样温度，控制生成随机性。值越高结果越随机。
        top_p (`float`): Nucleus 采样阈值，保留累计概率大于 top_p 的最小词集。
        top_k (`Optional[int]`): 限制每步仅从概率最高的前 K 个词中进行采样。
        repetition_penalty (`Optional[float]`): 重复惩罚因子 (大于 1.0 开启)，防止模型反复生成相同 Token。
        exclude_penalty_tokens (`Optional[List[int]]`): 在重复惩罚中需要被豁免的 Token IDs (如标点符号、换行)。
        suppress_tokens (`Optional[List[int]]`): 强行抑制不被生成的 Token IDs，将其 Logits 置为 -inf。
    """
    max_seq_len: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = 1.0
    exclude_penalty_tokens: Optional[List[int]] = None
    suppress_tokens: Optional[List[int]] = None


@dataclass(kw_only=True)
class PretrainConfig:
    """
    无监督预训练 (Pre-Training) 参数配置项。

    Args:
        gradient_accumulation_steps (`int`): 梯度累积步数，模拟更大的 Batch Size 以适应小显存。
        kd_config (`Optional[KDConfig]`): 知识蒸馏相关配置，若为 None 则不启用。
    """
    gradient_accumulation_steps: int = 1
    kd_config: Optional[KDConfig] = None


@dataclass(kw_only=True)
class SFTConfig:
    """
    监督微调 (Supervised Fine-Tuning) 参数配置项。

    Args:
        mask_prompt (`bool`): 是否使用 ignore_index (-100) Mask 掉输入中的 Prompt 部分，使模型只对回答计算 Loss。
        gradient_accumulation_steps (`int`): 梯度累积步数。
        kd_config (`Optional[KDConfig]`): 知识蒸馏相关配置。
        image_tags_file_dataset (`Optional[FileDataset]`): 多模态 SFT 场景下提供对应数据集的图像 Tag 映射。
        pixel_values_provider (`Optional[Callable]`): VLM 训练时，根据 image_tag 提供图像像素矩阵的回调。
        freeze_llm_model (`bool`): VLM 微调中是否冻结底座大模型，仅微调 Projector 投影层。
    """
    mask_prompt: bool = True
    gradient_accumulation_steps: int = 1
    kd_config: Optional[KDConfig] = None
    image_tags_file_dataset: Optional[FileDataset] = None
    pixel_values_provider: Optional[Callable[[list[str]], torch.Tensor]] = None
    freeze_llm_model: bool = False


@dataclass(kw_only=True)
class DPOConfig:
    """
    直接偏好优化 (Direct Preference Optimization) 训练配置。

    Args:
        ref_model_weights_path (Optional[str]): 参考模型 (Reference Model) 的初始化权重路径。
        mask_prompt (`bool`): 是否在 Chosen 和 Rejected 数据中屏蔽掉 Prompt 部分的损失。
        gradient_accumulation_steps (`int`): 梯度累积步数。
        loss_beta (`float`): DPO 的 KL 散度约束强度参数 (通常为 0.1)。值越大，模型越紧贴参考模型。
        loss_label_smoothing (`float`): DPO Loss 的标签平滑系数 (c-DPO 变体)。
        loss_ipo (`bool`): 是否采用 IPO (Identity Preference Optimization) 的 Loss 形式。
        nll_loss_coef (`Optional[float]`): 加入辅助的负对数似然 (NLL) 损失权重，缓解 DPO 生成质量退化。
    """
    ref_model_weights_path: Optional[str] = None
    mask_prompt: bool = True
    gradient_accumulation_steps: int = 1
    loss_beta: float
    loss_label_smoothing: float = 0.0
    loss_ipo: bool = False
    nll_loss_coef: Optional[float] = None


@dataclass(kw_only=True)
class PPOConfig:
    """
    近端策略优化 (Proximal Policy Optimization, PPO) 算法训练配置。

    Args:
        ppo_epochs (`int`): 在当前 Rollout 数据批次上，反复更新 Policy 和 Value 模型的次数。
        ppo_batch_size (`int`): PPO 内部计算 Loss 时的微批次 (Micro-batch) 大小。
        ref_model_weights_path (`Optional[str]`): PPO 计算 KL 惩罚奖励时参考模型的权重路径。
        value_model_weights_path (`Optional[str]`): Value (Critic) 模型的初始化权重路径。
        value_optim_config (`Optional[OptimConfig]`): 专门为 Value 模型配置独立的优化器及学习率。
        gradient_accumulation_steps (`int`): 梯度累积步数。
        gamma (`float`): 优势函数 (GAE) 中的折扣因子 (Discount Factor)，决定长期奖励的衰减。
        lam (`float`): GAE 中的 lambda 参数，权衡偏差与方差。
        clip_eps (`float`): PPO 的核心裁剪阈值，限制新旧策略更新的步长差距，防止更新崩溃。
        vf_coef (`float`): 总 Loss 中 Value Loss 的权重系数。
        kl_beta (`float`): 基于 KL 散度的初始惩罚奖励系数。
        kl_estimator (`str`): 计算近似 KL 散度的方法，支持 "k1" (log ratio) 或 "k3" (严格近似)。
        ptx_coef (`float`): 预训练数据 (PTX) Loss 的混合占比系数，用于缓解灾难性遗忘。
        missing_eos_penalty (`Optional[float]`): 针对模型未能正常生成 EOS (结束符) 的硬性奖励惩罚值。
        normalize_rewards (`bool`): 是否在喂给 GAE 前对环境 Reward 进行标准化。
        normalize_method (`str`): Reward 标准化方法，"RunningMeanStd" (流式均值方差) 或 "BatchStd" (当前批次方差)。
        whiten_rewards (`bool`): 是否对 GAE 计算后的优势 (Advantage) 进行白化处理。
        generate_config (`GenerateConfig`): PPO Rollout 交互生成数据时的解码策略。
    """
    ppo_epochs: int
    ppo_batch_size: int
    ref_model_weights_path: Optional[str] = None
    value_model_weights_path: Optional[str] = None
    value_optim_config: Optional[OptimConfig] = None
    gradient_accumulation_steps: int = 1
    gamma: float = 1.0
    lam: float = 0.95
    clip_eps: float = 0.1
    vf_coef: float = 0.5
    kl_beta: float = 0.02
    kl_estimator: str = 'k1'
    ptx_coef: float = 0.0
    missing_eos_penalty: Optional[float] = None
    normalize_rewards: bool = False
    normalize_method: str = 'RunningMeanStd'
    whiten_rewards: bool = False
    generate_config: GenerateConfig = field(default_factory=GenerateConfig)


@dataclass(kw_only=True)
class GRPOConfig:
    """
    组相对策略优化 (Group Relative Policy Optimization, DeepSeek V3 核心强化学习算法) 配置。

    Args:
        grpo_epochs (`int`): 同一批数据的复用训练次数。
        grpo_batch_size (`int`): 模型前向/反向的微批次大小。
        group_size (`int`): 对同一个 Prompt 并行生成多少个不同的答案，用于组内 Advantage 优势归一化计算。
        ref_model_weights_path (Optional[str]): GRPO 计算 KL 惩罚奖励时参考模型的权重路径。
        gradient_accumulation_steps (`int`): 梯度累积步数。
        loss_beta (`float`): KL 惩罚强度。在特定模式下(loss_importance_sampling_level=sequence) 可设为 0.0 改为隐式约束。
        loss_clip_eps (`float`): PPO 基础截断的下限 epsilon。
        loss_clip_eps_high (`Optional[float]`): 不对称裁剪中的上限 epsilon。
        loss_delta (`Optional[float]`): Advantage 权重的绝对上限阈值。
        loss_importance_sampling_level (`str`): 组相对优化的计算层级，支持 'token' 或 'sequence' 级截断。
        loss_type (`str`): GRPO 的 Loss 变体族，支持 'grpo', 'bnpo', 'dr_grpo', 'cispo', 'dapo', 'luspo', 'sapo', 'vespo' 等前沿算子。
        sapo_temperature_pos (`float`): SAPO/VESPO 针对正向优势的调节温度。
        sapo_temperature_neg (`float`): SAPO/VESPO 针对负向优势的调节温度。
        vespo_k_pos (`float`): VESPO 特定参数。
        vespo_lambda_pos (`float`): VESPO 特定参数。
        vespo_k_neg (`float`): VESPO 特定参数。
        vespo_lambda_neg (`float`): VESPO 特定参数。
        ptx_coef (`float`): 加入预训练监督数据的 Loss 混合权重系数。
        generate_config (`GenerateConfig`): Rollout 时的采样与生成策略。
    """
    grpo_epochs: int
    grpo_batch_size: int
    group_size: int = 12
    ref_model_weights_path: Optional[str] = None
    gradient_accumulation_steps: int = 1
    loss_beta: float = 0.04
    loss_clip_eps: float = 3e-4
    loss_clip_eps_high: Optional[float] = 4e-4
    loss_delta: Optional[float] = None
    loss_importance_sampling_level: str = 'token'
    loss_type: str = 'grpo'
    sapo_temperature_pos: float = 1.0
    sapo_temperature_neg: float = 1.0
    vespo_k_pos: float = 2.0
    vespo_lambda_pos: float = 3.0
    vespo_k_neg: float = 3.0
    vespo_lambda_neg: float = 2.0
    ptx_coef: float = 0.0
    generate_config: GenerateConfig = field(default_factory=GenerateConfig)


@dataclass(kw_only=True)
class TrainConfig:
    """
    全局训练入口参数配置项。控制训练的所有核心层级调度。

    Args:
        n_epochs (`int`): 全局数据集需要训练的 Epoch 轮数。
        batch_size (`int`): Global Batch Size (每个 GPU 每次 Data Loader 取出的数据条数)。
        model_config (`Union[ModelConfig, VLMConfig]`): LLM/VLM 底层模型的元配置定义。
        init_weights_path (`Optional[str]`): 初始化主干模型的权重路径。
        file_dataset (`FileDataset`): 用于加载训练数据的 DataSet 类实例。
        dataset_block_size (`int`): 序列截断长度。如果不传将取 Model 的 max_position_embedding。
        data_loader_config (`DataLoaderConfig`): PyTorch Dataloader 配置项（如 shuffle/worker）。
        optim_config (`OptimConfig`): 优化器 (Lion/Adam) 及学习率配置。
        ds_config (`DsConfig`): 掌控分布式底层的 DeepSpeed 引擎配置（含 ZeRO）。
        eval_config (`GenerateConfig`): 训练期间触发 Evaluation 测试集时的生成配置。
        save_interval (`int`): 指定多少个 global batch step 后触发一次 checkpoint 保存。
        eval_interval (`int`): 指定多少个 global batch step 后触发一次 checkpoint 保存。
        gradient_checkpointing (`bool`): 是否开启梯度检查点，如果开启且使用ds模式，会自动配置DsActivationCheckpointingConfig
        pretrain_config (`Optional[PretrainConfig]`): 使用基础 Trainer 时的配置组。
        sft_config (`Optional[SFTConfig]`): 使用 SFTTrainer 时的监督微调配置组。
        dpo_config (`Optional[DPOConfig]`): 使用 DPOTrainer 时的对齐微调配置组。
        ppo_config (`Optional[PPOConfig]`): 使用 PPOTrainer 时的强化学习微调配置组。
        grpo_config (`Optional[GRPOConfig]`): 使用 GRPOTrainer 时的核心组相关强化学习微调配置组。
    """
    n_epochs: int
    batch_size: int
    model_config: Union[ModelConfig, VLMConfig]
    init_weights_path: Optional[str] = None

    file_dataset: FileDataset
    dataset_block_size: int
    data_loader_config: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    optim_config: OptimConfig = field(default_factory=OptimConfig)
    ds_config: DsConfig = field(default_factory=DsConfig)

    eval_config: GenerateConfig = field(default_factory=GenerateConfig)
    save_interval: int = 100
    eval_interval: int = 100
    gradient_checkpointing: bool = False

    pretrain_config: Optional[PretrainConfig] = None
    sft_config: Optional[SFTConfig] = None
    dpo_config: Optional[DPOConfig] = None
    ppo_config: Optional[PPOConfig] = None
    grpo_config: Optional[GRPOConfig] = None

    def __post_init__(self):
        if self.gradient_checkpointing and self.ds_config is not None and self.ds_config.activation_checkpointing is None:
            self.ds_config.activation_checkpointing = DsActivationCheckpointingConfig()
        elif not self.gradient_checkpointing and self.ds_config is not None and self.ds_config.activation_checkpointing is not None:
            self.gradient_checkpointing = True

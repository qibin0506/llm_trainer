import os
import math
from abc import ABC, abstractmethod
from .tokenizer import Tokenizer
from .parallel import DsParallel, NoneParallel
from .log import Logger


_PARALLEL_TYPES = {
    'ds': DsParallel,
    'none': NoneParallel
}

_SUPPORT_DTYPE = ['auto', 'bf16', 'fp16', 'fp32']

class TrainerTools:
    def __init__(self):
        if not hasattr(TrainerTools, "_first_init"):
            TrainerTools._first_init = True

            self.parallel = self._new_parallel()
            self.tokenizer = Tokenizer()

            self.compute_dtype = os.environ.get('COMPUTE_DTYPE', 'auto').lower()
            if self.compute_dtype not in _SUPPORT_DTYPE:
                raise ValueError(f'DTYPE not in {_SUPPORT_DTYPE}')

            self.use_amp = (self.compute_dtype != 'fp32'
                            and (self.parallel.device_type != 'cpu')
                            and not isinstance(self.parallel, DsParallel))

            Logger.std_log(f'word_size={self.parallel.world_size}, use_amp={self.use_amp}')

    def _new_parallel(self):
        parallel_type = os.environ.get('PARALLEL_TYPE', 'none')
        Logger.std_log(f'parallel_type={parallel_type}')
        return _PARALLEL_TYPES[parallel_type]()

    def __new__(cls, *args, **kwargs):
        if not hasattr(TrainerTools, "_instance"):
            TrainerTools._instance = object.__new__(cls)

        return TrainerTools._instance


class FileDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx) -> str: ...


def estimate_data_size(
        file_dataset: FileDataset,
        block_size: int,
        type: str
) -> int:
    """
    估计数据集大小
    """
    data_size = 0
    files_count = len(file_dataset)

    if type == 'sft':
        from .dataset import SFTDataset
        for idx in range(files_count):
            dataset = SFTDataset(file_dataset[idx], block_size)
            data_size += len(dataset)
    elif type == 'dpo':
        from .dataset import DPODataset
        for idx in range(files_count):
            dataset = DPODataset(file_dataset[idx], block_size)
            data_size += len(dataset)
    elif type == 'grpo' or type == 'ppo':
        from .dataset import RLDataset
        for idx in range(files_count):
            dataset = RLDataset(file_dataset[idx])
            data_size += len(dataset)
    else:
        from .dataset import PretrainDataset
        for idx in range(files_count):
            dataset = PretrainDataset(
                file_dataset[idx],
                block_size,
                block_size
            )
            data_size += len(dataset)

    return data_size


def extract_policy_weights_from_ppo(model_config, ppo_weights):
    from llm_model import LlmModel
    from .ppo_trainer import PolicyAndValueModelWrapper, ValueModel

    policy_model = LlmModel(model_config)
    value_model = ValueModel(LlmModel(model_config))

    wrapper = PolicyAndValueModelWrapper(policy_model, value_model)
    wrapper.load_state_dict(ppo_weights)

    return wrapper.policy_model.state_dict()


def extract_value_weights_from_ppo(model_config, ppo_weights):
    from llm_model import LlmModel
    from .ppo_trainer import PolicyAndValueModelWrapper, ValueModel

    policy_model = LlmModel(model_config)
    value_model = ValueModel(LlmModel(model_config))

    wrapper = PolicyAndValueModelWrapper(policy_model, value_model)
    wrapper.load_state_dict(ppo_weights)

    return wrapper.value_model.state_dict()


def compute_lr_scheduler_steps(
        train_stage: str,
        epochs: int,
        all_data_size: int,
        batch_size: int,
        gradient_accumulation_steps: int,
        **kwargs
):
    world_size = TrainerTools().parallel.world_size

    # 基础 dataloader 的总 batch 数量（每个 GPU 上的 batch 数）
    dataloader_batches_per_gpu = epochs * (all_data_size // (batch_size * world_size))

    if train_stage in ['pretrain', 'midtrain', 'sft', 'dpo']:
        # DPO 和常规的 SFT/Pretrain 更新逻辑一致：直接在 dataloader batch 级别上做梯度累积
        train_batch_per_world = dataloader_batches_per_gpu / gradient_accumulation_steps
    elif train_stage == 'ppo':
        # PPO 算法特性：
        # - 数据加载：每次 dataloader 给出 batch_size 条数据进行 1 次 Rollout。
        # - 训练拆分：对 Rollout 数据训练 ppo_epochs 次，每次按 ppo_batch_size 拆分成 micro_batch 进行 forward+backward。
        # - 梯度累积：每 gradient_accumulation_steps 个 micro_batch 执行一次 step()。
        ppo_epochs = kwargs.get('ppo_epochs', 1)
        ppo_batch_size = kwargs.get('ppo_batch_size', 1)

        updates_per_dataloader_batch = (ppo_epochs * batch_size / ppo_batch_size) / gradient_accumulation_steps
        train_batch_per_world = dataloader_batches_per_gpu * updates_per_dataloader_batch
    elif train_stage == 'grpo':
        # GRPO 算法特性：
        # - 数据加载：每次 dataloader 给出 batch_size 个 prompt，内部生成 batch_size * group_size 条数据。
        # - 训练拆分：对这批扩增后的数据训练 grpo_epochs 次，按 grpo_batch_size 拆分为 micro_batch。
        # - 梯度累积：每 gradient_accumulation_steps 个 micro_batch 执行一次 step()。
        grpo_epochs = kwargs.get('grpo_epochs', 1)
        group_size = kwargs.get('group_size', 1)
        grpo_batch_size = kwargs.get('grpo_batch_size', 1)

        updates_per_dataloader_batch = (grpo_epochs * batch_size * group_size / grpo_batch_size) / gradient_accumulation_steps
        train_batch_per_world = dataloader_batches_per_gpu * updates_per_dataloader_batch
    else:
        train_batch_per_world = dataloader_batches_per_gpu / gradient_accumulation_steps

    train_batch_per_world = math.floor(train_batch_per_world)
    warmup_iters = int(0.1 * train_batch_per_world)

    max_warmup_iters = kwargs.get('max_warmup_iters', -1)
    if max_warmup_iters > -1:
        warmup_iters = min(warmup_iters, max_warmup_iters)

    cosine_annealing_batches = math.ceil(train_batch_per_world - warmup_iters)

    return warmup_iters, cosine_annealing_batches

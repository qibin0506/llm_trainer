from typing import Optional, Union, List, Sequence
from abc import ABC, abstractmethod
import os
import math

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
        type: str,
        return_per_file: bool = False
) -> Union[int, List[int]]:
    """
    估计数据集大小。

    Args:
        file_dataset (FileDataset): 文件数据集对象。
        block_size (int): 序列最大长度。
        type (str): 训练阶段类型 ('pretrain', 'sft', 'dpo', 'ppo', 'grpo')。
        return_per_file (bool): 若为 True，返回各文件独立的样本量列表 [len_f0, len_f1, ...]；
                               若为 False，返回总样本量 int（保持向后兼容）。
    """
    file_sizes = []
    files_count = len(file_dataset)

    if type == 'sft':
        from .dataset import SFTDataset
        for idx in range(files_count):
            dataset = SFTDataset(file_dataset[idx], block_size)
            file_sizes.append(len(dataset))
    elif type == 'dpo':
        from .dataset import DPODataset
        for idx in range(files_count):
            dataset = DPODataset(file_dataset[idx], block_size)
            file_sizes.append(len(dataset))
    elif type == 'grpo' or type == 'ppo':
        from .dataset import RLDataset
        for idx in range(files_count):
            dataset = RLDataset(file_dataset[idx])
            file_sizes.append(len(dataset))
    else:
        from .dataset import PretrainDataset
        for idx in range(files_count):
            dataset = PretrainDataset(
                file_dataset[idx],
                block_size,
                block_size
            )
            file_sizes.append(len(dataset))

    if return_per_file:
        return file_sizes
    return sum(file_sizes)


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
        all_data_size: Union[int, Sequence[int]],
        batch_size: int,
        gradient_accumulation_steps: int,
        warmup_rate: float,
        **kwargs
):
    world_size = TrainerTools().parallel.world_size
    batch_per_step_all_gpus = batch_size * world_size

    data_sizes_per_file = kwargs.get('data_sizes_per_file', None)
    if data_sizes_per_file is not None and isinstance(data_sizes_per_file, (list, tuple)):
        all_data_size = data_sizes_per_file

    if isinstance(all_data_size, (list, tuple)):
        file_sizes = list(all_data_size)
    else:
        file_sizes = [all_data_size]

    batches_per_file = [size // batch_per_step_all_gpus for size in file_sizes]

    if train_stage in ['pretrain', 'midtrain', 'sft', 'dpo']:
        steps_per_epoch = sum(
            math.ceil(batches / gradient_accumulation_steps)
            for batches in batches_per_file
        )
        train_batch_per_world = epochs * steps_per_epoch
    elif train_stage == 'ppo':
        ppo_epochs = kwargs.get('ppo_epochs', 1)
        ppo_batch_size = kwargs.get('ppo_batch_size', 1)

        micro_batches_per_ppo_epoch = math.ceil(batch_size / ppo_batch_size)
        updates_per_ppo_epoch = math.ceil(micro_batches_per_ppo_epoch / gradient_accumulation_steps)
        updates_per_dataloader_batch = ppo_epochs * updates_per_ppo_epoch

        dataloader_batches_per_gpu = epochs * sum(batches_per_file)
        train_batch_per_world = dataloader_batches_per_gpu * updates_per_dataloader_batch
    elif train_stage == 'grpo':
        grpo_epochs = kwargs.get('grpo_epochs', 1)
        group_size = kwargs.get('group_size', 1)
        grpo_batch_size = kwargs.get('grpo_batch_size', 1)

        micro_batches_per_grpo_epoch = math.ceil((batch_size * group_size) / grpo_batch_size)
        updates_per_grpo_epoch = math.ceil(micro_batches_per_grpo_epoch / gradient_accumulation_steps)
        updates_per_dataloader_batch = grpo_epochs * updates_per_grpo_epoch

        dataloader_batches_per_gpu = epochs * sum(batches_per_file)
        train_batch_per_world = dataloader_batches_per_gpu * updates_per_dataloader_batch
    else:
        steps_per_epoch = sum(
            math.ceil(batches / gradient_accumulation_steps)
            for batches in batches_per_file
        )
        train_batch_per_world = epochs * steps_per_epoch

    train_batch_per_world = math.ceil(train_batch_per_world)
    warmup_iters = int(warmup_rate * train_batch_per_world)

    max_warmup_iters = kwargs.get('max_warmup_iters', -1)
    if max_warmup_iters > -1:
        warmup_iters = min(warmup_iters, max_warmup_iters)

    cosine_annealing_batches = math.ceil(train_batch_per_world - warmup_iters)

    return warmup_iters, cosine_annealing_batches


def save_ds_weights_to_safetensors(input_path: str, output_path: str, dtype: Optional[str] = None):
    import torch
    from safetensors.torch import save_file
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    target_dtype = None
    if dtype is not None:
        dtype_str = dtype.lower()
        if dtype_str in ['fp16', 'float16']:
            target_dtype = torch.float16
        elif dtype_str in ['bf16', 'bfloat16']:
            target_dtype = torch.bfloat16
        elif dtype_str in ['fp32', 'float32']:
            target_dtype = torch.float32

    state_dict = get_fp32_state_dict_from_zero_checkpoint(input_path)

    if target_dtype is not None:
        clean_dict = {k: v.to(target_dtype).clone().contiguous() for k, v in state_dict.items()}
    else:
        clean_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}

    save_file(clean_dict, output_path)


def save_pt_weights_to_safetensors(input_path: str, output_path: str, dtype: Optional[str] = None):
    import torch
    from safetensors.torch import save_file

    target_dtype = None
    if dtype is not None:
        dtype_str = dtype.lower()
        if dtype_str in ['fp16', 'float16']:
            target_dtype = torch.float16
        elif dtype_str in ['bf16', 'bfloat16']:
            target_dtype = torch.bfloat16
        elif dtype_str in ['fp32', 'float32']:
            target_dtype = torch.float32

    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    if target_dtype is not None:
        clean_dict = {k: v.to(target_dtype).clone().contiguous() for k, v in state_dict.items()}
    else:
        clean_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}

    save_file(clean_dict, output_path)
from typing import Optional
import functools
import torch
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from .parallel import Parallel

class FsdpParallel(Parallel):
    def __init__(self):
        super().__init__()

    def process_model(
            self,
            model: nn.Module,
            ckpt_path: str,
            kwargs: Optional[dict] = None
    ) -> nn.Module:
        """
        :param model:
        :param ckpt_path:
        :param kwargs:
            "wrap_policy_num_params" int size_based_auto_wrap_policy的最小参数量
            "cpu_offload" bool 是否使用cpu卸载
            "offload_params" bool 是否卸载参数，在cpu_offload为True时生效
        :return:
        """

        model.to(self.device)

        # 先load state, 再compile，最后DDP
        self._load_ckpt(model, ckpt_path)

        if self._use_compile:
            model = torch.compile(model)

        if self._use_parallel:
            if 'transformer_layer_cls' in kwargs:
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls= {
                        kwargs['wrap_policy_num_params'],
                    }
                )
            elif 'wrap_policy_num_params' in kwargs:
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy,
                    min_num_params=kwargs['wrap_policy_num_params']
                )
            else:
                auto_wrap_policy = None

            if 'cpu_offload' in kwargs:
                offload_params = False
                if 'offload_params' in kwargs:
                    offload_params = kwargs['offload_params']

                # 选择配置 cpu_offload，以便在计算中不使用包装参数时将这些参数卸载到 CPU。
                # 这可以进一步提高内存效率，但代价是主机和设备之间的数据传输开销。
                cpu_offload = CPUOffload(offload_params=offload_params)
            else:
                cpu_offload = None

            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                mixed_precision = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    # Gradient communication precision.
                    reduce_dtype=torch.bfloat16,
                    # Buffer precision.
                    buffer_dtype=torch.bfloat16,
                )
            else:
                mixed_precision = None

            self.raw_model = model
            self.model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                cpu_offload=cpu_offload,
                device_id=torch.cuda.current_device()
            )
        else:
            self.model = model
            self.raw_model = model

        return self.model



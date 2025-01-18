from typing import Optional, Tuple
import functools
import torch
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
)

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    always_wrap_policy,
    enable_wrap,
    wrap,
)

from .parallel import Parallel

class FsdpParallel(Parallel):
    def __init__(self):
        super().__init__()

    def process(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            kwargs: Optional[dict] = None
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        :param model:
        :param optimizer:
        :param kwargs:
            "wrap_policy_num_params" int size_based_auto_wrap_policy的最小参数量
            "cpu_offload" bool 是否使用cpu卸载
            "offload_params" bool 是否卸载参数，在cpu_offload为True时生效
        :return:
        """

        model.to(self.device)

        if self._use_compile:
            model = torch.compile(model)

        if self._use_parallel:
            if 'transformer_layer_cls' in kwargs:
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=kwargs['transformer_layer_cls']
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

            # device_mesh = init_device_mesh("cuda", (self.world_size,))
            # self.model = FSDP(
            #     model,
            #     auto_wrap_policy=auto_wrap_policy,
            #     mixed_precision=mixed_precision,
            #     cpu_offload=cpu_offload,
            #     device_id=torch.cuda.current_device(),
            #     device_mesh=device_mesh
            # )

            self.model = FSDP(
                model,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision,
                cpu_offload=cpu_offload,
                device_id=torch.cuda.current_device(),
                process_group=None,
                # use_orig_params=True,
                # backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # bit faster async comms, bit higher memory
                # limit_all_gathers=False,
                # forward_prefetch=True,
            )
        else:
            self.model = model
            self.raw_model = model

        return self.model, optimizer



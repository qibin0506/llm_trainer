from typing import Optional
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from .tools import TrainerTools
from .parallel_ds import DsParallel
from .parallel_fsdp import FsdpParallel

def copy_model_params(
        _from: nn.Module,
        _to: Optional[nn.Module]
):
    """
        必须在所有rank上调用，非rank0, _to可以设置为None
    """
    if isinstance(TrainerTools().parallel, DsParallel):
        from .ds_model_params import get_ds_model_params
        state_dict = get_ds_model_params(_from, only_rank0=_to is None)
    elif isinstance(TrainerTools().parallel, FsdpParallel):
        from .fsdp_model_params import get_fsdp_model_params
        state_dict = get_fsdp_model_params(_from, only_rank0=_to is None)
    elif isinstance(_from, DDP):
        state_dict = _from.module.state_dict()
    else:
        state_dict = _from.state_dict()

    if _to and state_dict:
        _to.load_state_dict(state_dict)
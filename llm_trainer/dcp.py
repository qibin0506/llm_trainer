import os
from typing import Optional, Dict, Any
from torch import nn
from torch.optim import Optimizer
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

DEFAULT_DCP_DIR = "checkpoint"

class AppState(Stateful):
    def __init__(self, model: nn.Module, optimizer: Optional[Optimizer] = None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            'model': model_state_dict,
            'optim': optimizer_state_dict
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=state_dict['model_state_dict'],
            optim_state_dict=state_dict['optim_state_dict']
        )


def save_dcp(model: nn.Module, optimizer: Optional[Optimizer] = None):
    checkpoint_id = os.environ.get('DCP_DIR', DEFAULT_DCP_DIR)
    state_dict = {'app': AppState(model, optimizer)}
    dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_id)


def load_dcp(model: nn.Module, optimizer: Optional[Optimizer] = None):
    checkpoint_id = os.environ.get('DCP_DIR', DEFAULT_DCP_DIR)
    if os.path.exists(checkpoint_id):
        if isinstance(model, FSDP):
            # since no progress group is initialized, DCP will disable any collectives.
            state_dict = {'app': AppState(model, optimizer)}
            dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_id)
        else:
            state_dict = {
                "model_state_dict": model.state_dict(),
            }
            dcp.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_id,
            )
            model.load_state_dict(state_dict["model_state_dict"])

def convert_dcp_to_pth(pth_path: str):
    # convert dcp model to torch.save (assumes checkpoint was generated as above)
    dcp_to_torch_save(os.environ.get('DCP_DIR', DEFAULT_DCP_DIR), pth_path)

def convert_pth_to_dcp(pth_path: str):
    # converts the torch.save model back to DCP
    dcp_to_torch_save(pth_path, os.environ.get('DCP_DIR', DEFAULT_DCP_DIR))
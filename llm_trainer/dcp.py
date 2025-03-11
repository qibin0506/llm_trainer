import os
from typing import Optional, Dict, Any
from torch import nn
from torch.optim import Optimizer
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

DEFAULT_CHECKPOINT_DIR = "checkpoint"

class AppState(Stateful):
    def __init__(self, model: nn.Module, optimizer: Optimizer):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> Dict[str, Any]:
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        return {
            'model_state_dict': model_state_dict,
            'optim_state_dict': optimizer_state_dict
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        set_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            model_state_dict=state_dict['model_state_dict'],
            optim_state_dict=state_dict['optim_state_dict']
        )


def save_dcp(
        model: nn.Module,
        optimizer: Optimizer,
        suffix: Optional[str] = None
):
    checkpoint_id = os.environ.get('DIST_CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR)
    if suffix:
        checkpoint_id = f"{checkpoint_id}_{suffix}"

    state_dict = {'app': AppState(model, optimizer)}

    # fs_storage_writer = dcp.FileSystemWriter(checkpoint_id, overwrite=True)
    # dcp.save(state_dict=state_dict, storage_writer=fs_storage_writer)
    dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_id)


def load_dcp(
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        suffix: Optional[str] = None
):
    checkpoint_id = os.environ.get('DIST_CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR)
    if suffix:
        checkpoint_id = f"{checkpoint_id}_{suffix}"

    if os.path.exists(checkpoint_id):
        state_dict = {'app': AppState(model, optimizer)}
        # AppState帮助加载到state_dict中, 然后加载到model中
        dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_id)

        # if isinstance(model, FSDP):
        #     state_dict = {'app': AppState(model, optimizer)}
        #     # AppState帮助加载到state_dict中, 然后加载到model中
        #     dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_id)
        # else:
        #     state_dict = {"model_state_dict": model.state_dict()}
        #
        #     if optimizer:
        #         state_dict.update({'optim_state_dict': optimizer.state_dict()})
        #
        #     # since no progress group is initialized, DCP will disable any collectives.
        #     # 加载到state_dict中，然后通过model.load_state_dict加载到model中
        #     dcp.load(
        #         state_dict=state_dict,
        #         checkpoint_id=checkpoint_id,
        #     )
        #
        #     model.load_state_dict(state_dict["model_state_dict"])
        #     if optimizer:
        #         optimizer.load_state_dict(state_dict["optim_state_dict"])

def convert_dcp_to_pth(pth_path: str):
    dcp_path = os.environ.get('DIST_CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR)
    if os.path.exists(dcp_path):
        # convert dcp model to torch.save (assumes checkpoint was generated as above)
        dcp_to_torch_save(dcp_path, pth_path)

def convert_pth_to_dcp(pth_path: str):
    if os.path.exists(pth_path):
        # converts the torch.save model back to DCP
        torch_save_to_dcp(pth_path, os.environ.get('DIST_CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR))
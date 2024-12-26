import os
from typing import Optional, Union
import torch
from torch import nn
from torch.optim import Optimizer
# import torch.distributed.checkpoint as dcp
# from torch.distributed.checkpoint.stateful import Stateful
# from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
# from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from .train_tools import TrainerTools

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    StateDictType,
)

# https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

DEFAULT_CHECKPOINT_DIR = "checkpoint"

def save_checkpoint(model: nn.Module, optimizer: Optional[Optimizer] = None):
    if isinstance(model, FSDP):
        # 未经过测试 参考：https://doc.hfai.high-flyer.cn/haiscale/haiscale_fsdp.html
        # 是否使用rank0_only=True？
        with FSDP.summon_full_params(
                module=model,
                rank0_only=True,
                writeback=False,
                offload_to_cpu=True):
            if TrainerTools().parallel.is_main_process:
                states = model.state_dict()
                checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_DIR)
                ckpt = {'model_state_dict': states}
                torch.save(ckpt, checkpoint_name)

        # 经过测试过
        # if TrainerTools().parallel.is_main_process:
        #     states = model.state_dict()
        #     if TrainerTools().parallel.is_main_process:
        #         checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_DIR)
        #         ckpt = {'model_state_dict': states}
        #         torch.save(ckpt, checkpoint_name)
    else:
        if TrainerTools().parallel.is_main_process:
            checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_DIR)
            ckpt = {'model_state_dict': TrainerTools().parallel.raw_model.state_dict()}
            torch.save(ckpt, checkpoint_name)


    # if TrainerTools().parallel.is_main_process:
    #     checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_DIR)
    #     save_policy = {'rank0_only': True, 'pickle_module': torch.serialization}
    #     fsdp_to_save = model.module if hasattr(model, 'module') else model  # 获取原始模型
    #     torch.save({
    #         'model_state_dict': fsdp_to_save.state_dict()
    #     }, checkpoint_name, _use_new_zipfile_serialization=False)  # 解决一些可能的序列化问题
    #
    # # 同步所有进程
    # dist.barrier()
    # print(f'rank: {TrainerTools().parallel._local_rank}')


    # # use a barrier to make sure training is done on all ranks
    # dist.barrier()
    # state_dict = model.state_dict()
    # if TrainerTools().parallel.is_main_process:
    #     checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_DIR)
    #     ckpt = {'model_state_dict': state_dict}
    #     torch.save(ckpt, checkpoint_name)

def load_checkpoint(
        model: nn.Module,
        device: Optional[Union[torch.device, str]] = None,
        optimizer: Optional[Optimizer] = None
):
    checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_DIR)
    if os.path.exists(checkpoint_name):
        # 未经过测试，else的逻辑经过测试在fsdp下也没问题
        if isinstance(model, FSDP):
            with FSDP.summon_full_params(module=model):
                state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
        else:
            state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])

# class AppState(Stateful):
#     def __init__(self, model: nn.Module, optimizer: Optional[Optimizer] = None):
#         self.model = model
#         self.optimizer = optimizer
#
#     def state_dict(self) -> Dict[str, Any]:
#         model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
#         return {
#             'model': model_state_dict,
#             'optim': optimizer_state_dict
#         }
#
#     def load_state_dict(self, state_dict: Dict[str, Any]):
#         set_state_dict(
#             model=self.model,
#             optimizers=self.optimizer,
#             model_state_dict=state_dict['model'],
#             optim_state_dict=state_dict['optim']
#         )
#
#
# def save_checkpoint(model: nn.Module, optimizer: Optional[Optimizer] = None):
#     # use a barrier to make sure training is done on all ranks
#     dist.barrier()
#     state_dict = model.state_dict()
#     if TrainerTools().parallel.is_main_process:
#         checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_DIR)
#         torch.save(state_dict, checkpoint_name)
#
#
# def save_dcp(model: nn.Module, optimizer: Optional[Optimizer] = None):
#     checkpoint_id = os.environ.get('CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR)
#     state_dict = {'app': AppState(model, optimizer)}
#     dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_id)
#
#
# def load_dcp(model: nn.Module, optimizer: Optional[Optimizer] = None):
#     checkpoint_id = os.environ.get('CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR)
#     if os.path.exists(checkpoint_id):
#         if isinstance(model, FSDP):
#             # since no progress group is initialized, DCP will disable any collectives.
#             state_dict = {'app': AppState(model, optimizer)}
#             dcp.load(state_dict=state_dict, checkpoint_id=checkpoint_id)
#         else:
#             state_dict = {
#                 "model": model.state_dict(),
#             }
#             dcp.load(
#                 state_dict=state_dict,
#                 checkpoint_id=checkpoint_id,
#             )
#             model.load_state_dict(state_dict["model"])
#         # model.load_state_dict(state_dict["model"])
#
# def convert_dcp_to_pth(pth_path: str):
#     # convert dcp model to torch.save (assumes checkpoint was generated as above)
#     dcp_to_torch_save(os.environ.get('CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR), pth_path)
#
# def convert_pth_to_dcp(pth_path: str):
#     # converts the torch.save model back to DCP
#     dcp_to_torch_save(pth_path, os.environ.get('CHECKPOINT_DIR', DEFAULT_CHECKPOINT_DIR))
import os
from typing import Optional, Union
import torch
from torch import nn
from torch.optim import Optimizer
from .train_tools import TrainerTools
from .dcp import save_dcp, load_dcp, convert_dcp_to_pth

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html

DEFAULT_CHECKPOINT_NAME = "checkpoint.pth"

def save_checkpoint(model: nn.Module, optimizer: Optional[Optimizer] = None):
    if os.environ.get('ENABLE_DCP', '1') == '1':
        save_dcp(model, optimizer)
        return

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
                checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
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
            checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)
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
        optimizer: Optional[Optimizer] = None,
        device: Optional[Union[torch.device, str]] = None
):
    checkpoint_name = os.environ.get('CHECKPOINT_NAME', DEFAULT_CHECKPOINT_NAME)

    if os.environ.get('ENABLE_DCP', '1') == '1':
        if isinstance(model, FSDP):
            load_dcp(model, optimizer)
        else:
            # load_dcp方式在cpu上会报错，所以改为先将ckpt转换为pth，然后再加载pth
            # load_dcp(model, optimizer)
            if os.path.exists(checkpoint_name):
                os.remove(checkpoint_name)

            convert_dcp_to_pth(checkpoint_name)
            ckpt = torch.load(checkpoint_name, map_location=device, weights_only=True)
            model_state_dict = ckpt['app']['model_state_dict']
            model.load_state_dict(model_state_dict)
        return

    if os.path.exists(checkpoint_name):
        # 未经过测试，else的逻辑经过测试在fsdp下也没问题
        if isinstance(model, FSDP):
            with FSDP.summon_full_params(module=model):
                state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
        else:
            state_dict = torch.load(checkpoint_name, weights_only=True, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
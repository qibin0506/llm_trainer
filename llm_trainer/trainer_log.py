import os
import threading
import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from llama import LlamaConfig
from .generate_utils import generate
from .train_tools import TrainerTools
from .checkpoint import load_checkpoint


def _submit_gen_task(eval_model: torch.nn.Module, tag, prompt, max_position_embeddings, max_new_tokens):
    def task():
        save_dir = os.environ['SAVE_DIR']

        with open(f'{save_dir}gen_temp.txt', 'w') as tf:
            def write_temp(item):
                tf.write(item)
                tf.flush()
            
            # 当eval_model不是独立model时可以尝试这个
            # if isinstance(eval_model, FSDP):
            #     with FSDP.summon_full_params(module=eval_model, writeback=False, recurse=False):
            #         gen = generate(
            #             eval_model,
            #             prompt=prompt,
            #             max_position_embeddings=max_position_embeddings,
            #             max_new_tokens=max_new_tokens,
            #             # temperature=None,
            #             # k=None,
            #             # p=None,
            #             device='cpu',
            #             item_callback=lambda item: write_temp(item)
            #         )
                
            load_checkpoint(eval_model, device='cpu')
            gen = generate(
                eval_model,
                prompt=prompt,
                max_position_embeddings=max_position_embeddings,
                max_new_tokens=max_new_tokens,
                # temperature=None,
                # k=None,
                # p=None,
                device='cpu',
                item_callback=lambda item: write_temp(item)
            )

        with open(f'{save_dir}gen.txt', 'a') as f:
            f.write(f"{tag}, gen->{gen}\n")

    threading.Thread(target=task).start()


# def _save_model(state_dict):
#     save_dir = os.environ['SAVE_DIR']
#     if os.path.exists(f'{save_dir}modeling.pth'):
#         if os.path.exists(f'{save_dir}modeling_bak.pth'):
#             os.remove(f'{save_dir}modeling_bak.pth')
#
#         os.rename(f'{save_dir}modeling.pth', f'{save_dir}modeling_bak.pth')
#
#     ckpt = {'model': state_dict}
#
#     # if isinstance(TrainerTools().parallel, FsdpParallel):
#     #     # 如果是fsdp模式，则需要等所有gpu本轮都完毕后才能保存
#     #     # use a barrier to make sure training is done on all ranks
#     #     dist.barrier()
#     #
#     #     ckpt = {'model': TrainerTools().parallel.model.state_dict()}
#     # else:
#     #     ckpt = {'model': TrainerTools().parallel.raw_model.state_dict()}
#
#     torch.save(ckpt, f'{save_dir}modeling.pth')


# def _get_model_state_dict():
#     if (isinstance(TrainerTools().parallel, FsdpParallel)
#             or isinstance(TrainerTools().parallel, DdpParallel)):
#         state_dict = get_model_state_dict(TrainerTools().parallel.model)
#     else:
#         state_dict = TrainerTools().parallel.raw_model.state_dict()
#
#     return state_dict


def on_batch(eval_model, epoch, batch, loss, need_update_grad, prompt, llama_config: LlamaConfig):
    if TrainerTools().parallel.is_main_process:
        if (batch + 1) % 100 == 0:
            save_dir = os.environ['SAVE_DIR']
            print(f"epoch: {epoch}, batch: {batch}")
            with open(f'{save_dir}batch.txt', 'a') as f:
                f.write(f"epoch: {epoch}, batch: {batch}, loss: {loss}, need_update_grad: {need_update_grad}\n")

            _submit_gen_task(
                eval_model,
                tag=f'sign:batch/epoch:{epoch}/batch:{batch}',
                prompt=prompt,
                max_position_embeddings=llama_config.max_position_embeddings,
                max_new_tokens=100
            )


def on_file(eval_model, epoch, batch, prompt, llama_config: LlamaConfig):
    if TrainerTools().parallel.is_main_process:
        _submit_gen_task(
            eval_model,
            tag=f'sign:file/epoch:{epoch}/batch:{batch}',
            prompt=prompt,
            max_position_embeddings=llama_config.max_position_embeddings,
            max_new_tokens=100
        )


def on_exception(e, epoch, batch):
    if isinstance(e, KeyboardInterrupt):
        if TrainerTools().parallel.is_main_process:
            # ddp_helper.raw_model.train()
            # save_model()
            exit(0)
    elif isinstance(e, torch.OutOfMemoryError):
        if TrainerTools().parallel.is_main_process:
            save_dir = os.environ['SAVE_DIR']
            with open(f'{save_dir}batch.txt', 'a') as f:
                exception_file = e.__traceback__.tb_frame.f_globals["__file__"]
                exception_line = e.__traceback__.tb_lineno
                f.write(f"epoch: {epoch}, batch: {batch}, {e} at {exception_file} line {exception_line}\n")
    else:
        raise e


def on_epoch(eval_model, epoch, loss_accumulation, all_batch, need_update_grad, prompt, llama_config: LlamaConfig):
    if TrainerTools().parallel.is_main_process:

        # test_loss = test_loop(model, test_data_loader)
        print(f'train_loss: {loss_accumulation.detach().item() / all_batch}')

        save_dir = os.environ['SAVE_DIR']
        with open(f'{save_dir}batch.txt', 'a') as f:
            f.write(
                f"epoch: {epoch},"
                f" loss: {loss_accumulation.detach().item() / all_batch},"
                f" need_update_grad:{need_update_grad}\n")

        _submit_gen_task(
            eval_model,
            tag=f'sign:epoch/epoch:{epoch}',
            prompt=prompt,
            max_position_embeddings=llama_config.max_position_embeddings,
            max_new_tokens=100
        )

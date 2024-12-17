import torch
from .utils import generate, TrainConfig
import os
import threading


def submit_gen_task(eval_model, sign, epoch, batch, state_dict, prompt, max_new_tokens):
    def task():
        save_dir = os.environ['SAVE_DIR']

        with open(f'{save_dir}gen_temp.txt', 'w') as tf:
            def write_temp(item):
                tf.write(item)
                tf.flush()

            eval_model.load_state_dict(state_dict)
            gen = generate(eval_model, prompt,
                            max_new_tokens, device='cpu', item_callback=lambda item: write_temp(item))
        with open(f'{save_dir}gen.txt', 'a') as f:
            f.write(f"{sign}/epoch:{epoch},batch:{batch},gen:{gen}\n")

    threading.Thread(target=task).start()


def save_model():
    save_dir = os.environ['SAVE_DIR']
    if os.path.exists(f'{save_dir}modeling.pth'):
        if os.path.exists(f'{save_dir}modeling_bak.pth'):
            os.remove(f'{save_dir}modeling_bak.pth')

        os.rename(f'{save_dir}modeling.pth', f'{save_dir}modeling_bak.pth')

    ckpt = {'model': TrainConfig().ddp_helper.raw_model.state_dict()}
    torch.save(ckpt, f'{save_dir}modeling.pth')


def on_batch(eval_model, epoch, batch, loss, need_update_grad, prompt):
    if TrainConfig().ddp_helper.is_main_process():
        if (batch + 1) % 100 == 0:
            save_dir = os.environ['SAVE_DIR']
            print(f"epoch: {epoch}, batch: {batch}")
            with open(f'{save_dir}batch.txt', 'a') as f:
                f.write(f"epoch: {epoch}, batch: {batch}, loss: {loss}, need_update_grad: {need_update_grad}\n")

            save_model()
            submit_gen_task(eval_model, 'batch', epoch, batch, TrainConfig().ddp_helper.raw_model.state_dict(), prompt, 256)


def on_file(eval_model, epoch, batch, prompt):
    if TrainConfig().ddp_helper.is_main_process():
        submit_gen_task(eval_model, 'file', epoch, batch, TrainConfig().ddp_helper.raw_model.state_dict(), prompt, 256)


def on_exception(e, epoch, batch):
    if isinstance(e, KeyboardInterrupt):
        if TrainConfig().ddp_helper.is_main_process():
            # ddp_helper.raw_model.train()
            # save_model()
            exit(0)
    elif isinstance(e, Exception):
        if TrainConfig().ddp_helper.is_main_process():
            save_dir = os.environ['SAVE_DIR']
            with open(f'{save_dir}batch.txt', 'a') as f:
                exception_file = e.__traceback__.tb_frame.f_globals["__file__"]
                exception_line = e.__traceback__.tb_lineno
                f.write(f"epoch: {epoch}, batch: {batch}, {e} at {exception_file} line {exception_line}\n")


def on_epoch(eval_model, epoch, loss_accumulation, all_batch, need_update_grad, prompt):
    if TrainConfig().ddp_helper.is_main_process():
        save_model()

        # test_loss = test_loop(model, test_data_loader)
        print(f'train_loss: {loss_accumulation.detach().item() / all_batch}')

        save_dir = os.environ['SAVE_DIR']
        with open(f'{save_dir}batch.txt', 'a') as f:
            f.write(
                f"epoch: {epoch}, loss: {loss_accumulation.detach().item() / all_batch}, need_update_grad:{need_update_grad}\n")

        submit_gen_task(eval_model, 'epoch', epoch, -1, TrainConfig().ddp_helper.raw_model.state_dict(), prompt, 256)
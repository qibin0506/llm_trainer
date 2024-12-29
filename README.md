# llm_trainer

``` python
from llm_trainer import TrainerTools, TrainArgs, FsdpArgs, DataLoaderArgs
from llama import LlamaConfig
from llama import LlamaDecoderLayer
import os
from glob import glob

def init_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKENIZERS_TYPE'] = '1'
    os.environ['TOKEN_DIR'] = './tokens/'

    os.environ['SAVE_DIR'] = './log/'

    os.environ['PARALLEL_TYPE'] = 'fsdp'  # or 'ddp'

    os.environ['ENABLE_DCP'] = '1'
    os.environ['CHECKPOINT_NAME'] = 'ckpt.pth'
    os.environ['DCP_DIR'] = 'ckpt_dir'


def get_config():
    return LlamaConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=2048,
        intermediate_size=11008,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=8,
        max_position_embeddings=512
    )


def get_train_config(is_sft: bool):
    desire_batch_size = 32
    real_batch_size = 8
    assert desire_batch_size % real_batch_size == 0
    gradient_accumulation_steps = desire_batch_size // real_batch_size

    train_args = TrainArgs(
        n_epochs=1,
        batch_size=1,
        llama_config=get_config(),
        is_sft=False,
        all_data_size=64131,
        all_files=glob('./data/pretrain/*.pkl'),
        gradient_accumulation_steps=gradient_accumulation_steps,
        fsdp_args=FsdpArgs(
            transformer_layer_cls={ LlamaDecoderLayer },
            wrap_policy_num_params=20000,
            cpu_offload=True,
            offload_params=True
        ),
        data_loader_args=DataLoaderArgs(
            data_loader_pin_memory=True,
            data_loader_num_workers=4,
            data_loader_shuffle=False,
            data_loader_drop_last=True
        )
    )

    if is_sft:
        train_args.n_epochs = 2
        train_args.batch_size = real_batch_size
        train_args.is_sft = True
        train_args.all_data_size = 10000
        train_args.all_files = glob('./data/train/sft.pkl')
    else:
        train_args.n_epochs = 1
        train_args.batch_size = real_batch_size
        train_args.is_sft = False
        train_args.all_data_size = 806244
        train_args.all_files = glob('./data/skypile/raw/*.pkl')

    return train_args

```

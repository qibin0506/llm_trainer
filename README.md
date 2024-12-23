# llm_trainer

```python
from lm_trainer import TrainerTools
from llama import LlamaConfig
from llm_trainer import TrainArgs, FsdpArgs, DataLoaderArgs
from llm_trainer import train_fn
import os
from glob import glob


def init_env():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    os.environ['TOKENIZERS_TYPE'] = '1'
    os.environ['TOKEN_DIR'] = './tokens/'

    os.environ['SAVE_DIR'] = './'

    os.environ['PARALLEL_TYPE'] = 'fsdp' # or 'ddp'


def get_config():
    config = LlamaConfig(
        vocab_size=TrainerTools().tokenizer.vocab_size,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        max_position_embeddings=1024
    )

    return config


def get_train_args(is_pretrain: bool) -> TrainArgs:
    train_args = TrainArgs(
        n_epochs=0,
        batch_size=0,
        llama_config=get_config(),
        is_sft=False,
        all_data_size=0,
        all_files=[],
        gradient_accumulation_steps=32,
        fsdp_args=FsdpArgs(
            transformer_layer_cls=None,
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

    if is_pretrain:
        train_args.n_epochs = 5
        train_args.batch_size = 5
        train_args.is_sft = False
        train_args.all_data_size = 92230
        train_args.all_files = glob('./data/train/pretrain.pkl')
    else:
        train_args.n_epochs = 5
        train_args.batch_size = 5
        train_args.is_sft = True
        train_args.all_data_size = 9968
        train_args.all_files = glob('./data/train/sft.pkl')

    return train_args



if __name__ == '__main__':
    init_env()

    train_fn(
        train_args=get_train_args(is_pretrain=True),
        prompt_on_batch="半瓶香水 涩涩的香味\n往事在泪水中回味\n",
        prompt_on_epoch="某年某月的某一天\n就象一张破碎的脸\n"
    )

```

import os
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from pytorch.llm.llama import KVCache
from .tokenizer import Tokenizer
from .ddp import DDPHelper


class TrainConfig:
    def __init__(self):
        if not hasattr(TrainConfig, "_first_init"):
            TrainConfig._first_init = True

            # 梯度积累步数
            self.gradient_accumulation_steps = self._get_env_int('GRADIENT_ACCUMULATION_STEPS', 0)

            self.ddp_helper = DDPHelper()
            self.tokenizer = Tokenizer(self._get_env_int('TOKENIZERS_TYPE', 0))

            self.use_amp = 'cuda' in self.ddp_helper.device
            self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

            self.bot_token = self.tokenizer.encode_to_token('[BOT]', unsqueeze=False, covert_tensor=False)[0]

    def _get_env_int(self, name, default_value: int) -> int:
        try:
            value = os.environ[name]
            return int(value)
        except:
            return default_value

    def __new__(cls, *args, **kwargs):
        if not hasattr(TrainConfig, "_instance"):
            TrainConfig._instance = object.__new__(cls)

        return TrainConfig._instance


# def get_llama_config():
#     return LlamaConfig(
#         vocab_size=tokenizer.vocab_size,
#         hidden_size=2048,
#         intermediate_size=5632,
#         num_hidden_layers=22,
#         num_attention_heads=32,
#         num_key_value_heads=8,
#         max_position_embeddings=2048)


# def get_llama_config():
#     return LlamaConfig(
#         vocab_size=tokenizer.vocab_size,
#         hidden_size=1024,
#         intermediate_size=4096,
#         num_hidden_layers=22,
#         num_attention_heads=32,
#         num_key_value_heads=8,
#         max_position_embeddings=1024)


def pretrain_padding_fn(batch_data):
    inputs = pad_sequence(batch_data, batch_first=True, padding_value=TrainConfig().tokenizer.pad)
    # crossEntropy默认的ignore_index是-100
    labels = pad_sequence(batch_data, batch_first=True, padding_value=-100)

    return inputs, labels


def sft_padding_fn(batch_data):
    """
     如果是sft，则不计算prompt部分的loss, 例如：
    logits: [USER]:你好[BOT]:我好[SEP]
    labels: [USER]:你好[BOT]:我好[SEP]

    shift_logits: [USER]:你好[BOT]:我好
    shift_labels: :你好[BOT]:我好[SEP]

    mask_labels: mask mask mask mask:我好[SEP]
        * [BOT]后的:暂时不考虑，后续可以把prompt里的：去掉
        * mask=-100和pad一样
    """
    inputs, labels = pretrain_padding_fn(batch_data)
    batch_size = len(labels)
    batch_bot_idx = torch.nonzero(torch.eq(labels, TrainConfig().bot_token), as_tuple=True)[1]

    for batch in range(batch_size):
        bot_idx = batch_bot_idx[batch].item() + 1
        labels[batch, :bot_idx] = torch.tensor([-100] * bot_idx)

    return inputs, labels


# def pretrain_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#     logits = logits.reshape(-1, logits.shape[-1])
#     targets = labels.reshape(-1)
#
#     return F.cross_entropy(logits, targets, ignore_index=-100)


def calc_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # logits shape (batch, seq_len, vocab_size)
    # labels shape (batch, seq_len)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    logits = shift_logits.reshape(-1, logits.shape[-1])
    targets = shift_labels.reshape(-1)

    return F.cross_entropy(logits, targets, ignore_index=-100)


def generate_text(
        model,
        tokens,
        ctx_len,
        max_new_tokens,
        temperature,
        device,
        topk=None,
        token_item_callback=None
):
    use_kv_cache = True

    enable_autocast = 'cuda' in device
    kv_cache: KVCache = None
    generate_tokens = tokens.clone()

    for _ in range(max_new_tokens):
        t = tokens[:, -ctx_len:]
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=TrainConfig().dtype, enabled=enable_autocast):
                # logits (batch, seq_len, vocab_size)
                logits, kv_cache = model(t, past_key_values=kv_cache, use_cache=use_kv_cache)

        # (batch, vocab_size)
        logits = logits[:, -1, :]
        # 抑制[UNK]输出
        logits[..., TrainConfig().tokenizer.unk] = torch.tensor(-torch.inf)

        if topk is not None:
            topk_logits, _ = torch.topk(logits, k=topk)
            min_val: torch.Tensor = topk_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(-torch.inf).to(device), logits)

        if temperature > 0:
            logits /= temperature
            prob = logits.softmax(dim=-1)
            # 返回下标
            next_token = torch.multinomial(prob, num_samples=1)
        else:
            # 返回下标
            next_token = logits.argmax(dim=-1, keepdim=True)

        if token_item_callback is not None:
            token_item_callback(next_token)

        if use_kv_cache:
            tokens = next_token
            generate_tokens = torch.cat((generate_tokens, next_token), dim=-1)
        else:
            tokens = torch.cat((tokens, next_token), dim=-1)

        if next_token.item() == TrainConfig().tokenizer.eot:
            break

    return tokens if not use_kv_cache else generate_tokens


def generate(
        model,
        *,
        prompt,
        max_position_embeddings,
        max_new_tokens,
        temperature=0.6,
        topk=3,
        device=None,
        item_callback=None,
):
    model.eval()

    if item_callback is not None:
        token_item_callback = lambda token: item_callback(TrainConfig().tokenizer.decode_to_text(token))
    else:
        token_item_callback = None

    device = TrainConfig().ddp_helper.device if device is None else device
    encoded = TrainConfig().tokenizer.encode_to_token(prompt).to(device)
    output = generate_text(model, encoded,
                           max_position_embeddings, max_new_tokens,
                           temperature, device, topk, token_item_callback)
    decoded = TrainConfig().tokenizer.decode_to_text(output)

    return decoded

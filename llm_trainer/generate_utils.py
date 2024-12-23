from typing import Union, Optional
import torch
from llama import KVCache
from .utils import TrainerTools


def _suppress_warper(logits: torch.Tensor, suppress_tokens: list[int]) -> torch.Tensor:
    """
    抑制特殊token输出
    :param logits:
    :param suppress_tokens:
    :return:
    """
    suppress_tokens = torch.tensor(suppress_tokens, device=logits.device)
    vocab_tensor = torch.arange(logits.shape[-1], device=logits.device)
    suppress_token_mask = torch.isin(vocab_tensor, suppress_tokens)
    logits = torch.where(suppress_token_mask, -float("inf"), logits)

    return logits


def _temperature_warper(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    应用temperature
    :param logits:
    :param temperature:
    :return:
    """
    logits = logits / temperature
    return logits


def _top_k_warper(logits: torch.Tensor, k: float, device: Union[str, torch.device, int] = None) -> torch.Tensor:
    """
    top k采样
    :param logits:
    :param k:
    :param device:
    :return:
    """
    topk_logits, _ = torch.topk(logits, k=k)
    min_val: torch.Tensor = topk_logits[:, -1]
    logits = torch.where(logits < min_val, torch.tensor(-torch.inf).to(device), logits)
    return logits


def _top_p_warper(logits: torch.Tensor, p: float, min_tokens_to_keep: int = 1) -> torch.Tensor:
    """
    top p 核采样
    :param logits:
    :param p:
    :param min_tokens_to_keep:
    :return:
    """
    # 正序排列 eg: [0.1, 0.2, 0.3]
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    # cumsum求和, 每一个元素的值都是与之前元素的求和
    # 例如：torch.cumsum(torch.tensor([[0.1, 0.2, 0.3]]), dim=-1) 结果是: [0.1, 0.3, 0.6]
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # 删除累积概率<=1-p的部分, 因为cumulative_probs是正序排列的，所以要用1-p
    # 例如：
    #   假设 p=0.9，并且经过排序和计算后，我们有以下的 cumulative_probs
    #       cumulative_probs = [0.1, 0.3, 0.7, 0.92, 0.98]
    #       那么 (1 - p) 就是 0.1
    #       执行 cumulative_probs <= (1 - p) 后，得到的 sorted_indices_to_remove 就是
    #       sorted_indices_to_remove = [True, False, False, False, False]
    #       这意味着，累积概率小于等于 0.1 的词（也就是第一个词）应该被移除
    #   为什么是 (1 - p)？
    #       这里使用 (1 - p) 的原因是为了方便后续的处理。在实际的代码中，
    #       通常会将 sorted_indices_to_remove 向右移动一位，并将第一个元素设置为 False。
    #       这样做是为了保留至少一个词，即使第一个词的概率非常小。
    #       通过使用 (1 - p)，我们可以直接使用 cumulative_probs 进行比较，而不需要额外的步骤来处理第一个词
    sorted_indices_to_remove = cumulative_probs <= (1 - p)
    # 保证至少有min_tokens_to_keep个token保留
    # 例如:
    #   sorted_indices_to_remove=[True, True, True]，min_tokens_to_keep=1时
    #   该操作后sorted_indices_to_remove=[True, True, False]
    sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0
    # 下面步骤是将排序后确定要删除的元素index映射回非排序的元素index
    #  scatter函数根据 index 中提供的索引，将 src 中的值复制到 tensor 中。
    # 举例说明, 假设我们有一个 batch，词汇表大小为 5，并且有以下数据
    #   sorted_indices = [[2, 0, 4, 1, 3]] (排序后的索引)
    #   sorted_indices_to_remove = [[False, True, False, True, False]] (排序后的移除掩码)
    # 执行sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove) 后，得到的 indices_to_remove 将是
    #   indices_to_remove = [[True, True, False, False, False]]
    indices_to_remove = sorted_indices_to_remove.scatter(1, index=sorted_indices, src=sorted_indices_to_remove)

    # 将需要移除的元素的值设置为-inf
    scores_processed = logits.masked_fill_(indices_to_remove, -float("Inf"))

    return scores_processed


def generate_text(
        model: torch.nn.Module,
        *,
        tokens: torch.Tensor,
        max_position_embeddings: int,
        max_new_tokens: int,
        temperature: Optional[float],
        k: Optional[int],
        p: Optional[float],
        suppress_tokens: Optional[list[int]] = None,
        device: Union[str, torch.device, int],
        token_item_callback
):
    """
    :param model:
    :param tokens:
    :param max_position_embeddings:
    :param max_new_tokens:
    :param temperature: 设置None不不生效temperature
    :param k: top k参数，设置为None或者0不生效topk
    :param p: top p参数，设置为None不生效top p
    :param suppress_tokens: 要抑制的tokens
    :param device:
    :param token_item_callback:

    如果内容质量底，需要减小temperature、k、p
    如果temperature很大但内容单一，需要增大k、p
    """
    use_kv_cache = True

    enable_autocast = 'cuda' in device
    kv_cache: Optional[KVCache] = None
    generate_tokens = tokens.clone()

    for _ in range(max_new_tokens):
        t = tokens[:, -max_position_embeddings:]
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=TrainerTools().dtype, enabled=enable_autocast):
                # logits (batch, seq_len, vocab_size)
                logits, kv_cache = model(t, past_key_values=kv_cache, use_cache=use_kv_cache)

        # (batch, vocab_size)
        logits = logits[:, -1, :]
        # 抑制特殊token输出
        if suppress_tokens is not None and len(suppress_tokens) != 0:
            logits = _suppress_warper(logits, suppress_tokens)

        multinomial = False
        if temperature is not None and temperature > 0:
            multinomial = True
            logits = _temperature_warper(logits, temperature)

        if k is not None and k != 0:
            logits = _top_k_warper(logits, k, device)

        if p is not None and p < 1:
            logits = _top_p_warper(logits, p)

        if multinomial:
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

        if next_token.item() == TrainerTools().tokenizer.eot:
            break

    return tokens if not use_kv_cache else generate_tokens


def generate(
        model: torch.nn.Module,
        *,
        prompt: str,
        max_position_embeddings: int,
        max_new_tokens: int,
        temperature: Optional[float] = 1.0,
        k: Optional[int] = 50,
        p: Optional[float] = 1.0,
        suppress_tokens: Optional[list[int]] = None,
        device: Union[str, torch.device, int] = None,
        item_callback = None,
):
    model.eval()

    if item_callback is not None:
        token_item_callback = lambda token: item_callback(TrainerTools().tokenizer.decode_to_text(token))
    else:
        token_item_callback = None

    device = TrainerTools().parallel.device if device is None else device
    encoded_tokens = TrainerTools().tokenizer.encode_to_token(prompt).to(device)

    output = generate_text(
        model=model,
        tokens=encoded_tokens,
        max_position_embeddings=max_position_embeddings,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        k=k,
        p=p,
        device=device,
        suppress_tokens=suppress_tokens,
        token_item_callback=token_item_callback
    )

    decoded = TrainerTools().tokenizer.decode_to_text(output)

    return decoded
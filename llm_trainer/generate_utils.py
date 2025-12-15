from typing import Union, Optional, List
import torch
from llm_model import VlmModel, KVCache
from .tools import TrainerTools
from .utils import (
    autocast,
    batch_repeat_image_tok,
    calc_position_ids
)


def _suppress_warper(logits: torch.Tensor, suppress_tokens: List[int]) -> torch.Tensor:
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


def _top_k_warper(logits: torch.Tensor, k: int, device: Union[str, torch.device, int] = None) -> torch.Tensor:
    """
    top k采样
    :param logits:
    :param k:
    :param device:
    :return:
    """
    # [batch, k]
    topk_logits, _ = torch.topk(logits, k=k)
    # []
    min_val: torch.Tensor = topk_logits[:, -1]
    logits = torch.where(logits < min_val.unsqueeze(-1), torch.tensor(-torch.inf).to(device), logits)
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


def _generate(
        model: torch.nn.Module,
        *,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float],
        k: Optional[int],
        p: Optional[float],
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int]
):
    """
    :param model:
    :param tokens:
    :param max_new_tokens:
    :param temperature: 设置None不不生效temperature
    :param k: top k参数，设置为None或者0不生效topk
    :param p: top p参数，设置为None不生效top p
    :param suppress_tokens: 要抑制的tokens
    :param device:

    如果内容质量底，需要减小temperature、k、p
    如果temperature很大但内容单一，需要增大k、p
    """
    use_kv_cache = True

    # 确保输入维度是 [Batch, Seq]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    attention_mask = torch.ones_like(tokens, device=device, dtype=torch.long)

    if isinstance(model, VlmModel):
        tokens = batch_repeat_image_tok(tokens, tokens_per_image)

    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    kv_cache: Optional[KVCache] = None
    generate_tokens = tokens.clone()

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            t = tokens
            with autocast(TrainerTools().parallel.device_type):
                result = model(
                    t,
                    attention_mask=attention_mask,
                    past_key_values=kv_cache,
                    use_cache=use_kv_cache,
                    pixel_values=pixel_values
                )

                # logits (batch, seq_len, vocab_size)
                logits = result['logits']
                kv_cache = result['past_key_values']

            # (batch, vocab_size)
            logits = logits[:, -1, :]

            # 抑制特殊token输出
            if suppress_tokens and len(suppress_tokens) != 0:
                logits = _suppress_warper(logits, suppress_tokens)

            multinomial = False
            if temperature and temperature > 0:
                multinomial = True
                logits = _temperature_warper(logits, temperature)

            if k and k != 0:
                logits = _top_k_warper(logits, k, device)

            if p and 0 < p <= 1:
                logits = _top_p_warper(logits, p)

            if multinomial:
                prob = logits.softmax(dim=-1)
                # 返回下标
                next_token = torch.multinomial(prob, num_samples=1)
            else:
                # 返回下标
                next_token = logits.argmax(dim=-1, keepdim=True)

            # token, is_full_result
            yield next_token, False

            if use_kv_cache:
                tokens = next_token
                generate_tokens = torch.cat((generate_tokens, next_token), dim=-1)
            else:
                tokens = torch.cat((tokens, next_token), dim=-1)

            # [关键修复] 更新 mask：追加 1，让 Position ID 继续增长
            new_mask_bit = torch.ones((tokens.shape[0], 1), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, new_mask_bit), dim=-1)

            if next_token.item() == TrainerTools().tokenizer.end:
                break

    # token, is_full_result
    yield tokens if not use_kv_cache else generate_tokens, True


def _streaming_generate(
        model: torch.nn.Module,
        *,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int,
        temperature: Optional[float] = 1.0,
        k: Optional[int] = None,
        p: Optional[float] = None,
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int] = None
):
    device = TrainerTools().parallel.device if not device else device

    if isinstance(prompt, torch.Tensor):
        encoded_tokens = prompt.to(device)
    else:
        encoded_tokens = TrainerTools().tokenizer.encode(prompt, unsqueeze=True, covert_tensor=True).to(device)

    generate_text_iterator = _generate(
        model=model,
        tokens=encoded_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        k=k,
        p=p,
        pixel_values=pixel_values,
        tokens_per_image=tokens_per_image,
        suppress_tokens=suppress_tokens,
        device=device
    )

    for (token, is_full_result) in generate_text_iterator:
        yield token, is_full_result


def streaming_generate(
        model: torch.nn.Module,
        *,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int,
        temperature: Optional[float] = 1.0,
        k: Optional[int] = None,
        p: Optional[float] = None,
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int] = None,
        return_token: bool = False
):
    text_iterator = _streaming_generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        k=k,
        p=p,
        pixel_values=pixel_values,
        tokens_per_image=tokens_per_image,
        suppress_tokens=suppress_tokens,
        device=device
    )

    for (token, is_full_result) in text_iterator:
        if not is_full_result:
            if return_token:
                yield token.squeeze(0)
            else:
                yield TrainerTools().tokenizer.decode(token.squeeze(0))


def generate(
        model: torch.nn.Module,
        *,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int,
        temperature: Optional[float] = 1.0,
        k: Optional[int] = None,
        p: Optional[float] = None,
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int] = None,
        return_token: bool = False
):
    text_iterator = _streaming_generate(
        model=model,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        k=k,
        p=p,
        suppress_tokens=suppress_tokens,
        pixel_values=pixel_values,
        tokens_per_image=tokens_per_image,
        device=device
    )

    for (token, is_full_result) in text_iterator:
        if is_full_result:
            if return_token:
                return token.squeeze(0)
            else:
                return TrainerTools().tokenizer.decode(token.squeeze(0))

    return None


def batch_generate(
        model: torch.nn.Module,
        *,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float] = None,
        k: Optional[int] = None,
        p: Optional[float] = None,
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int],
        return_logits: bool = True
):
    use_kv_cache = True
    end_token = TrainerTools().tokenizer.end
    pad_token_id = TrainerTools().tokenizer.pad

    if isinstance(model, VlmModel):
        tokens = batch_repeat_image_tok(tokens, tokens_per_image)

    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    orig_tokens = tokens.clone()
    full_attention_mask = attention_mask.clone()

    # 初始化 position_ids，处理 left padding
    position_ids = calc_position_ids(full_attention_mask)

    kv_cache: Optional[KVCache] = None
    batch_size = tokens.shape[0]

    # 预分配最大长度，避免循环中 cat 造成内存碎片
    generated_tokens_buffer = torch.full(
        (batch_size, max_new_tokens),
        pad_token_id,
        dtype=torch.long,
        device=device
    )

    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    current_tokens = tokens

    padded_logits = None
    actual_gen_len = 0

    pad_token_tensor = torch.tensor(pad_token_id, device=device, dtype=torch.long)

    with torch.inference_mode():
        for i in range(max_new_tokens):
            if done.all():
                break

            actual_gen_len = i + 1

            if current_tokens.dtype != torch.long:
                current_tokens = current_tokens.long()

            if kv_cache is None:
                current_position_ids = position_ids
            else:
                # 下一个位置ID基于当前mask序列的最后一个有效位置
                # 如果kv_cache有效，当前token是上一步生成的，位置是前一个位置+1
                current_position_ids = position_ids[:, -1:] + 1
                position_ids = torch.cat((position_ids, current_position_ids), dim=-1)

            with autocast(TrainerTools().parallel.device_type):
                result = model(
                    current_tokens,
                    attention_mask=full_attention_mask,
                    position_ids=current_position_ids,
                    past_key_values=kv_cache,
                    use_cache=use_kv_cache,
                    pixel_values=pixel_values
                )
                logits = result['logits']
                kv_cache = result['past_key_values']

            logits = logits[:, -1, :]

            if return_logits:
                if padded_logits is None:
                    vocab_size = logits.shape[-1]
                    padded_logits = torch.zeros(
                        (batch_size, max_new_tokens, vocab_size),
                        dtype=logits.dtype,
                        device=device
                    )
                padded_logits[:, i, :] = logits

            if suppress_tokens:
                logits = _suppress_warper(logits, suppress_tokens)

            multinomial = False
            if temperature and temperature > 0:
                multinomial = True
                logits = _temperature_warper(logits, temperature)
            if k and k != 0:
                logits = _top_k_warper(logits, k, device)
            if p and 0 < p <= 1:
                logits = _top_p_warper(logits, p)

            if multinomial:
                prob = logits.softmax(dim=-1)
                next_token_active = torch.multinomial(prob, num_samples=1)
            else:
                next_token_active = logits.argmax(dim=-1, keepdim=True)

            next_token = torch.where(
                done.unsqueeze(1),
                pad_token_tensor,
                next_token_active
            )

            generated_tokens_buffer[:, i] = next_token.squeeze(-1)

            new_done = (next_token.squeeze(-1) == end_token)
            done = done | new_done

            current_tokens = next_token

            new_mask = (~done).long().to(full_attention_mask.dtype)
            full_attention_mask = torch.cat((full_attention_mask, new_mask.unsqueeze(-1)), dim=-1)

    final_generated_tokens = generated_tokens_buffer[:, :actual_gen_len]

    if padded_logits is not None:
        final_padded_logits = padded_logits[:, :actual_gen_len, :]
    else:
        final_padded_logits = None

    final_full_sequences = torch.cat((orig_tokens, final_generated_tokens), dim=1)

    return final_full_sequences, final_padded_logits

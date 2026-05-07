from typing import Union, Optional, List
import torch
from llm_model import VlmModel, KVCache
from .tools import TrainerTools
from .utils import (
    autocast,
    batch_repeat_image_tok,
    calc_position_ids
)


def _repetition_penalty_warper(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float,
        exclude_tokens: Optional[List[int]] = None
) -> torch.Tensor:
    if penalty == 1.0:
        return logits

    logits = logits.clone()
    valid_exclude =[]

    # 暂存需要排除惩罚的特殊 token 的原始 logits
    if exclude_tokens is not None and len(exclude_tokens) > 0:
        valid_exclude =[t for t in exclude_tokens if 0 <= t < logits.shape[-1]]
        if valid_exclude:
            saved_logits = logits[:, valid_exclude].clone()
        else:
            exclude_tokens = None

    score = torch.gather(logits, 1, input_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, input_ids, score)

    if exclude_tokens is not None and len(exclude_tokens) > 0 and valid_exclude:
        logits[:, valid_exclude] = saved_logits

    return logits


def _suppress_warper(
        logits: torch.Tensor,
        suppress_tokens: List[int]
) -> torch.Tensor:
    """
    抑制特殊token输出
    :param logits:
    :param suppress_tokens:
    :return:
    """
    if suppress_tokens:
        logits[..., suppress_tokens] = -float("inf")

    return logits


def _temperature_warper(
        logits: torch.Tensor,
        temperature: float
) -> torch.Tensor:
    """
    应用temperature
    :param logits:
    :param temperature:
    :return:
    """
    logits = logits / temperature
    return logits


def _top_k_warper(
        logits: torch.Tensor,
        top_k: int,
        device: Union[str,
        torch.device, int] = None
) -> torch.Tensor:
    """
    top k采样
    :param logits:
    :param top_k:
    :param device:
    :return:
    """
    # [batch, top_k]
    top_k = min(top_k, logits.shape[-1])
    topk_logits, _ = torch.topk(logits, k=top_k)
    # []
    min_val: torch.Tensor = topk_logits[:, -1]
    logits.masked_fill_(logits < min_val.unsqueeze(-1), -float("inf"))

    return logits


def _top_p_warper(
        logits: torch.Tensor,
        top_p: float,
        min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """
    top p 核采样
    :param logits:
    :param top_p:
    :param min_tokens_to_keep:
    :return:
    """
    # 正序排列 eg: [0.1, 0.2, 0.3]
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    # cumsum求和, 每一个元素的值都是与之前元素的求和
    # 例如：torch.cumsum(torch.tensor([[0.1, 0.2, 0.3]]), dim=-1) 结果是: [0.1, 0.3, 0.6]
    cumulative_probs = sorted_logits.float().softmax(dim=-1).cumsum(dim=-1)
    # 删除累积概率<=1-top_p的部分, 因为cumulative_probs是正序排列的，所以要用1-top_p
    # 例如：
    #   假设 top_p=0.9，并且经过排序和计算后，我们有以下的 cumulative_probs
    #       cumulative_probs = [0.1, 0.3, 0.7, 0.92, 0.98]
    #       那么 (1 - top_p) 就是 0.1
    #       执行 cumulative_probs <= (1 - top_p) 后，得到的 sorted_indices_to_remove 就是
    #       sorted_indices_to_remove = [True, False, False, False, False]
    #       这意味着，累积概率小于等于 0.1 的词（也就是第一个词）应该被移除
    #   为什么是 (1 - top_p)？
    #       这里使用 (1 - top_p) 的原因是为了方便后续的处理。在实际的代码中，
    #       通常会将 sorted_indices_to_remove 向右移动一位，并将第一个元素设置为 False。
    #       这样做是为了保留至少一个词，即使第一个词的概率非常小。
    #       通过使用 (1 - top_p)，我们可以直接使用 cumulative_probs 进行比较，而不需要额外的步骤来处理第一个词
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
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
    scores_processed = logits.masked_fill(indices_to_remove, -float("Inf"))

    return scores_processed


def _generate(
        model: torch.nn.Module,
        *,
        tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: Optional[float],
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: Optional[float] = 1.0,
        exclude_penalty_tokens: Optional[List[int]] = None,
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
    :param top_k: top k参数，设置为None或者0不生效top k
    :param top_p: top p参数，设置为None不生效top p
    :param repetition_penalty: 重复惩罚参数，设置为1.0或者None不生效
    :param suppress_tokens: 要抑制的tokens
    :param device:
    """
    use_kv_cache = True

    special_tokens = list(TrainerTools().tokenizer.get_special_tokens_dict().values())
    if exclude_penalty_tokens is not None:
        special_tokens.extend(exclude_penalty_tokens)
    special_tokens = list(set(special_tokens))

    # 确保输入维度是 [Batch, Seq]
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    assert tokens.shape[0] == 1

    if isinstance(model, VlmModel):
        tokens, _ = batch_repeat_image_tok(tokens, tokens_per_image)

    pad_token_id = TrainerTools().tokenizer.pad
    attention_mask = (tokens != pad_token_id).to(device=device, dtype=torch.long)

    batch_size = tokens.shape[0]
    prompt_len = tokens.shape[1]

    kv_cache: Optional[KVCache] = None
    if use_kv_cache:
        # Prompt Length + Max Generation Length
        total_capacity = prompt_len + max_new_tokens
        kv_cache = KVCache(max_capacity=total_capacity)

    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    # 提前分配流式生成的全局大 Buffer
    full_sequence_buffer = torch.full(
        (batch_size, prompt_len + max_new_tokens),
        pad_token_id,
        dtype=torch.long,
        device=device
    )
    full_sequence_buffer[:, :prompt_len] = tokens

    full_attention_mask = torch.zeros(
        (batch_size, prompt_len + max_new_tokens),
        dtype=attention_mask.dtype,
        device=device
    )
    full_attention_mask[:, :prompt_len] = attention_mask

    current_tokens = tokens
    actual_gen_len = 0

    with torch.inference_mode():
        for i in range(max_new_tokens):
            actual_gen_len = i + 1
            current_attention_mask = full_attention_mask[:, :prompt_len + i]

            with autocast(TrainerTools().parallel.device_type):
                result = model(
                    current_tokens,
                    attention_mask=current_attention_mask,
                    past_key_values=kv_cache,
                    use_cache=use_kv_cache,
                    pixel_values=pixel_values
                )

                # logits (batch, seq_len, vocab_size)
                logits = result['logits']

            # (batch, vocab_size)
            logits = logits[:, -1, :]

            # 抑制特殊token输出
            if suppress_tokens and len(suppress_tokens) != 0:
                logits = _suppress_warper(logits, suppress_tokens)

            # 重复性惩罚
            if repetition_penalty and repetition_penalty != 1.0:
                current_context = full_sequence_buffer[:, prompt_len:prompt_len + i]
                logits = _repetition_penalty_warper(
                    logits,
                    current_context,
                    repetition_penalty,
                    exclude_tokens=special_tokens
                )

            multinomial = False
            if temperature and temperature > 0:
                multinomial = True
                logits = _temperature_warper(logits, temperature)

            if top_k and top_k != 0:
                logits = _top_k_warper(logits, top_k, device)

            if top_p and 0 < top_p <= 1:
                logits = _top_p_warper(logits, top_p)

            if multinomial:
                prob = logits.float().softmax(dim=-1)
                if torch.isnan(prob).any():
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token = torch.multinomial(prob, num_samples=1)
            else:
                # 返回下标
                next_token = logits.argmax(dim=-1, keepdim=True)

            # token, is_full_result
            yield next_token, False

            full_sequence_buffer[:, prompt_len + i] = next_token.squeeze(-1)
            if use_kv_cache:
                current_tokens = next_token
            else:
                # 如果不用 KV Cache，模型需要每次传入前文全量 Tokens
                # 直接通过切片获取全局 Buffer，这里也不会触发显存拷贝！
                current_tokens = full_sequence_buffer[:, :prompt_len + i + 1]

            full_attention_mask[:, prompt_len + i] = 1
            if next_token.item() == TrainerTools().tokenizer.end:
                break

    final_full_sequences = full_sequence_buffer[:, :prompt_len + actual_gen_len]
    yield final_full_sequences, True


def _streaming_generate(
        model: torch.nn.Module,
        *,
        prompt: Union[str, torch.Tensor],
        max_new_tokens: int,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = 1.0,
        exclude_penalty_tokens: Optional[List[int]] = None,
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
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        exclude_penalty_tokens=exclude_penalty_tokens,
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
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = 1.0,
        exclude_penalty_tokens: Optional[List[int]] = None,
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
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        exclude_penalty_tokens=exclude_penalty_tokens,
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
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = 1.0,
        exclude_penalty_tokens: Optional[List[int]] = None,
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
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        exclude_penalty_tokens=exclude_penalty_tokens,
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
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = 1.0,
        exclude_penalty_tokens: Optional[List[int]] = None,
        pixel_values: Optional[torch.Tensor] = None,
        tokens_per_image: int = -1,
        suppress_tokens: Optional[List[int]] = None,
        device: Union[str, torch.device, int],
        return_logits: bool = True
):
    use_kv_cache = True
    end_token = TrainerTools().tokenizer.end
    pad_token_id = TrainerTools().tokenizer.pad

    special_tokens = list(TrainerTools().tokenizer.get_special_tokens_dict().values())
    if exclude_penalty_tokens is not None:
        special_tokens.extend(exclude_penalty_tokens)
    special_tokens = list(set(special_tokens))

    if isinstance(model, VlmModel):
        tokens, attention_mask = batch_repeat_image_tok(tokens, tokens_per_image, attention_mask)

    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    orig_tokens = tokens.clone()
    full_attention_mask = attention_mask.clone()

    # 初始化 position_ids，处理 left padding
    position_ids = calc_position_ids(full_attention_mask)

    kv_cache: Optional[KVCache] = None
    batch_size = tokens.shape[0]
    prompt_len = orig_tokens.shape[1]

    if use_kv_cache:
        # Prompt Length + Max Generation Length
        total_capacity = prompt_len + max_new_tokens
        kv_cache = KVCache(max_capacity=total_capacity)

    # 直接分配一个涵盖 全局 (Prompt + Generate) 的终极 Buffer
    full_sequence_buffer = torch.full(
        (batch_size, prompt_len + max_new_tokens),
        pad_token_id,
        dtype=torch.long,
        device=device
    )
    # 预先将 prompt 写入 Buffer 的开头
    full_sequence_buffer[:, :prompt_len] = orig_tokens

    full_attention_mask_buffer = torch.zeros(
        (batch_size, prompt_len + max_new_tokens),
        dtype=attention_mask.dtype,
        device=device
    )
    full_attention_mask_buffer[:, :prompt_len] = attention_mask

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

            current_attention_mask = full_attention_mask_buffer[:, :prompt_len + i]

            if kv_cache is None:
                current_position_ids = calc_position_ids(current_attention_mask)
            else:
                if i == 0:
                     current_position_ids = position_ids
                else:
                     current_position_ids = position_ids[:, -1:] + 1
                     position_ids = torch.cat((position_ids, current_position_ids), dim=-1)

            with autocast(TrainerTools().parallel.device_type):
                result = model(
                    current_tokens,
                    attention_mask=current_attention_mask,
                    position_ids=current_position_ids,
                    past_key_values=kv_cache,
                    use_cache=use_kv_cache,
                    pixel_values=pixel_values
                )
                logits = result['logits']

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

            if repetition_penalty and repetition_penalty != 1.0:
                current_context = full_sequence_buffer[:, prompt_len:prompt_len + i]
                logits = _repetition_penalty_warper(
                    logits,
                    current_context,
                    repetition_penalty,
                    exclude_tokens=special_tokens
                )

            multinomial = False
            if temperature and temperature > 0:
                multinomial = True
                logits = _temperature_warper(logits, temperature)

            if top_k and top_k != 0:
                logits = _top_k_warper(logits, top_k, device)

            if top_p and 0 < top_p <= 1:
                logits = _top_p_warper(logits, top_p)

            if multinomial:
                prob = logits.float().softmax(dim=-1)
                if torch.isnan(prob).any():
                    next_token_active = logits.argmax(dim=-1, keepdim=True)
                else:
                    next_token_active = torch.multinomial(prob, num_samples=1)
            else:
                next_token_active = logits.argmax(dim=-1, keepdim=True)

            next_token = torch.where(
                done.unsqueeze(1),
                pad_token_tensor,
                next_token_active
            )

            full_sequence_buffer[:, prompt_len + i] = next_token.squeeze(-1)

            new_done = (next_token.squeeze(-1) == end_token)
            done = done | new_done

            if use_kv_cache:
                current_tokens = next_token
            else:
                current_tokens = full_sequence_buffer[:, :prompt_len + i + 1]

            new_mask = (~done).long().to(full_attention_mask_buffer.dtype)
            full_attention_mask_buffer[:, prompt_len + i] = new_mask

    final_full_sequences = full_sequence_buffer[:, :prompt_len + actual_gen_len]

    if padded_logits is not None:
        final_padded_logits = padded_logits[:, :actual_gen_len, :]
    else:
        final_padded_logits = None

    return final_full_sequences, final_padded_logits
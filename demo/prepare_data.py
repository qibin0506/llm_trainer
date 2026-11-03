import os
import sys
import json
import urllib.request
from typing import List, Dict, Optional, Union

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
TOKEN_DIR = os.path.join(CURRENT_DIR, "data", "tokenizer")
os.environ["TOKEN_DIR"] = TOKEN_DIR


def ensure_raw_shakespeare() -> str:
    """Locate or download tiny_shakespeare text."""
    raw_path = os.path.join(CURRENT_DIR, "data", "shakespeare.txt")
    if os.path.exists(raw_path):
        return raw_path

    sibling_path = os.path.abspath(
        os.path.join(PROJECT_ROOT, "..", "LLMs-from-scratch", "ch05", "shakespeare.txt")
    )
    if os.path.exists(sibling_path):
        print(f"Copying shakespeare.txt from {sibling_path} to {raw_path}...")
        with open(sibling_path, "r", encoding="utf-8") as src, open(raw_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        return raw_path

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading tiny_shakespeare from {url}...")
    urllib.request.urlretrieve(url, raw_path)
    return raw_path


def build_nano_shakespeare_tokenizer(raw_text: str, token_dir: str):
    """
    Builds a character-level tokenizer matching Andrej Karpathy's nanoGPT (data/shakespeare_char/prepare.py).
    Captures all unique characters from tiny_shakespeare plus special control tokens required by llm_trainer:
    - 15 special tokens: </s>, <pad>, <unk>, <user>, <assistant>, <think>, </think>, <answer>, </answer>, <system>, etc.
    - 65 characters from Shakespeare.
    - Total vocab size: 80 tokens (compact, zero garbage/multilingual tokens, rapid convergence).
    """
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import BPE

    chars = sorted(list(set(raw_text)))
    special_tokens = [
        "</s>", "<pad>", "<unk>", "<user>", "<assistant>",
        "<think>", "</think>", "<answer>", "</answer>",
        "<system>", "<image>", "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>"
    ]

    vocab = {st: i for i, st in enumerate(special_tokens)}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    tok = Tokenizer(BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tok.add_special_tokens(special_tokens)
    tok.decoder = decoders.Fuse()

    os.makedirs(token_dir, exist_ok=True)
    tok.save(os.path.join(token_dir, "tokenizer.json"))
    with open(os.path.join(token_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "add_bos_token": False,
            "add_eos_token": False,
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "tokenizer_class": "PreTrainedTokenizerFast"
        }, f, indent=2, ensure_ascii=False)
    print(f"Created nanoGPT Shakespeare character tokenizer at {token_dir} (vocab size: {len(vocab)})")


def get_demo_model_config() -> "ModelConfig":
    """
    nanoGPT-inspired character-level Shakespeare Model (~0.61M parameters):
    - vocab_size=80 (from character tokenizer)
    - hidden_size=128, intermediate_size=256
    - 4 transformer layers, 4 attention heads (2 kv heads, head_dim=32)
    - max_position_embeddings=512 (expanded context window for character models)
    - tie_word_embeddings=False (safe for DeepSpeed ZeRO-3)
    """
    from llm_trainer import TrainerTools
    from llm_model import ModelConfig, RoPEConfig

    vocab_size = TrainerTools().tokenizer.vocab_size
    return ModelConfig(
        vocab_size=vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        attention_implementation="sdpa",
        use_qk_norm=True,
        tie_word_embeddings=False,
        rope_config=RoPEConfig(rope_theta=10000.0),
    )


def build_pretrain_data(raw_text: str, out_path: str, count: Optional[int] = None):
    """
    Chunk the entire Shakespeare corpus into natural verse paragraphs for pretraining,
    preserving natural linebreaks (\n) and formatting as in nanoGPT.
    """
    lines = raw_text.split("\n")
    chunks = []
    current_chunk = []
    current_len = 0
    for line in lines:
        current_chunk.append(line)
        current_len += len(line) + 1
        if current_len >= 220:
            chunk_text = "\n".join(current_chunk).strip()
            if len(chunk_text) > 30:
                chunks.append(chunk_text)
            current_chunk = []
            current_len = 0
    if current_chunk:
        chunk_text = "\n".join(current_chunk).strip()
        if len(chunk_text) > 30:
            chunks.append(chunk_text)

    selected = chunks[:count] if count is not None else chunks
    with open(out_path, "w", encoding="utf-8") as f:
        for p in selected:
            f.write(json.dumps({"text": p}, ensure_ascii=False) + "\n")
    print(f"Created Pretrain dataset at {out_path} ({len(selected)} samples from full corpus)")


def build_sft_data(raw_text: str, out_path: str, count: Optional[int] = None):
    """
    Build conversational dialogue pairs across the entire Shakespeare corpus,
    preserving character speaker names and authentic dramatic responses.
    """
    lines = raw_text.split("\n")
    dialogues: List[Dict[str, str]] = []
    current_speaker = None
    current_speech = []

    for line in lines:
        line_s = line.strip()
        if ":" in line_s and not line.startswith(" ") and not line.startswith("\t"):
            parts = line_s.split(":", 1)
            speaker = parts[0].strip()
            speech = parts[1].strip()
            if current_speaker and current_speech:
                dialogues.append({"speaker": current_speaker, "text": "\n".join(current_speech).strip()})
            current_speaker = speaker
            current_speech = [speech] if speech else []
        elif current_speaker:
            if line_s:
                current_speech.append(line_s)
            else:
                if current_speech:
                    dialogues.append({"speaker": current_speaker, "text": "\n".join(current_speech).strip()})
                    current_speaker = None
                    current_speech = []

    if current_speaker and current_speech:
        dialogues.append({"speaker": current_speaker, "text": "\n".join(current_speech).strip()})

    total_pairs = len(dialogues) - 1
    pair_count = min(count, total_pairs) if count is not None else total_pairs
    sft_samples = []
    for i in range(pair_count):
        d1 = dialogues[i]
        d2 = dialogues[i + 1]
        if len(d1["text"]) < 5 or len(d2["text"]) < 5:
            continue
        t1 = d1["text"][:100]
        t2 = d2["text"][:100]
        conv = [
            {"role": "user", "content": f"{d1['speaker']}: {t1}"},
            {"role": "assistant", "content": f"{d2['speaker']}: {t2}"}
        ]
        sft_samples.append(conv)

    with open(out_path, "w", encoding="utf-8") as f:
        for conv in sft_samples:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
    print(f"Created SFT dataset at {out_path} ({len(sft_samples)} samples from full corpus)")


def build_dpo_data(raw_text: str, out_path: str, count: Optional[int] = None):
    """
    Build DPO preference pairs:
    - Chosen: authentic Shakespearean dialogue continuation
    - Rejected: corrupt / modern slang rejections
    """
    lines = raw_text.split("\n")
    dialogues: List[Dict[str, str]] = []
    current_speaker = None
    current_speech = []

    for line in lines:
        line_s = line.strip()
        if ":" in line_s and not line.startswith(" ") and not line.startswith("\t"):
            parts = line_s.split(":", 1)
            speaker = parts[0].strip()
            speech = parts[1].strip()
            if current_speaker and current_speech:
                dialogues.append({"speaker": current_speaker, "text": "\n".join(current_speech).strip()})
            current_speaker = speaker
            current_speech = [speech] if speech else []
        elif current_speaker:
            if line_s:
                current_speech.append(line_s)
            else:
                if current_speech:
                    dialogues.append({"speaker": current_speaker, "text": "\n".join(current_speech).strip()})
                    current_speaker = None
                    current_speech = []

    if current_speaker and current_speech:
        dialogues.append({"speaker": current_speaker, "text": "\n".join(current_speech).strip()})

    rejections = [
        "Nah whatever, I do not care about this whole thing.",
        "That sounds like too much trouble, lets just go home.",
        "I am an AI assistant and cannot engage with this uprising.",
        "Error: no Shakespearean wisdom found here today.",
        "Why speak like this? Please speak normal modern English.",
        "Whatever you say, but I am skipping duty today.",
        "I completely disagree and shall refuse your command.",
        "LOL no way, not interested in this drama.",
    ]

    dpo_data = []
    total_pairs = len(dialogues) - 1
    pair_count = min(count, total_pairs) if count is not None else total_pairs

    for i in range(pair_count):
        d1 = dialogues[i]
        d2 = dialogues[i + 1]
        if len(d1["text"]) < 5 or len(d2["text"]) < 5:
            continue
        t1 = d1["text"][:100]
        chosen_text = f"{d2['speaker']}: {d2['text'][:100]}"
        rejected_text = rejections[i % len(rejections)]

        user_msg = {"role": "user", "content": f"{d1['speaker']}: {t1}"}
        sample = {
            "chosen": [
                user_msg,
                {"role": "assistant", "content": chosen_text}
            ],
            "rejected": [
                user_msg,
                {"role": "assistant", "content": rejected_text}
            ]
        }
        dpo_data.append(sample)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in dpo_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Created DPO dataset at {out_path} ({len(dpo_data)} samples from full corpus)")


def build_rl_data(raw_text: str, out_path: str, count: Optional[int] = 500):
    """Build RL prompt queries across dialogue openers in Shakespeare."""
    lines = [line.strip() for line in raw_text.split("\n") if len(line.strip()) > 20 and ":" in line.strip()]
    total_lines = len(lines)
    if count is not None and count < total_lines:
        step = max(1, total_lines // count)
        selected_lines = lines[::step][:count]
    else:
        selected_lines = lines

    rl_data = []
    for full_line in selected_lines:
        parts = full_line.split(":", 1)
        speaker = parts[0].strip()
        speech = parts[1].strip() if len(parts) > 1 else ""
        words = speech.split()
        if len(words) >= 4:
            prompt_speech = " ".join(words[:4])
            gt_answer = " ".join(words[4:])
        else:
            prompt_speech = speech
            gt_answer = speech

        sample = {
            "prompt": [
                {"role": "user", "content": f"{speaker}: {prompt_speech} ..."}
            ],
            "answer": gt_answer
        }
        rl_data.append(sample)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in rl_data:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Created RL dataset at {out_path} ({len(rl_data)} samples sampled across full corpus)")


def save_base_checkpoint():
    """Save an initial base checkpoint for reference model and init_weights."""
    import torch
    from llm_model import LlmModel

    ckpt_dir = os.path.join(CURRENT_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    base_model_path = os.path.join(ckpt_dir, "base_model.pt")

    config = get_demo_model_config()
    print("Initializing base test model and saving to checkpoints/base_model.pt...")
    model = LlmModel(config)
    torch.save(model.state_dict(), base_model_path)
    print(f"Base model checkpoint saved at {base_model_path} ({sum(p.numel() for p in model.parameters()):,} parameters)")


def main():
    os.makedirs(os.path.join(CURRENT_DIR, "data"), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, "logs"), exist_ok=True)

    print("=== Step 1: Ensuring tiny_shakespeare text ===")
    raw_path = ensure_raw_shakespeare()
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("=== Step 2: Building nanoGPT character tokenizer (vocab_size=80) ===")
    build_nano_shakespeare_tokenizer(raw_text, TOKEN_DIR)

    print("=== Step 3: Building training datasets ===")
    build_pretrain_data(raw_text, os.path.join(CURRENT_DIR, "data", "pretrain.jsonl"))
    build_sft_data(raw_text, os.path.join(CURRENT_DIR, "data", "sft.jsonl"))
    build_dpo_data(raw_text, os.path.join(CURRENT_DIR, "data", "dpo.jsonl"))
    build_rl_data(raw_text, os.path.join(CURRENT_DIR, "data", "rl.jsonl"))

    print("=== Step 4: Generating initial base model checkpoint ===")
    save_base_checkpoint()
    print("nanoGPT Shakespeare data and model preparation complete!")


if __name__ == "__main__":
    main()

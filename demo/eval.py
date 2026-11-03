import os
import sys
import json
import math
import argparse
from typing import List, Dict, Tuple, Optional, Any

# Ensure local packages llm_model and llm_trainer can be imported
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

from common import (
    init_demo_env,
    get_demo_model_config,
    CURRENT_DIR,
    demo_reward_func,
    format_chat_prompt,
    get_demo_eval_prompts,
)

init_demo_env(stage="eval")

import torch
import torch.nn.functional as F
from llm_model import LlmModel
from llm_trainer import TrainerTools
from llm_trainer.generate_utils import generate


SHAKESPEARE_KEYWORDS = [
    "citizen", "speak", "hear", "people", "rome", "corn", "marcius", "caius",
    "resolved", "famish", "senate", "enemy", "lord", "king", "patricians",
    "shall", "thee", "thou", "verdict", "alive", "death", "die", "hath",
    "doth", "art", "well", "good", "sir", "all", "our", "men", "peace", "yours"
]

BENCHMARK_PROMPTS = [
    ("Prompt 1 (Dialogue)", "First Citizen:\nBefore we proceed any further, hear me speak."),
    ("Prompt 2 (Dialogue)", "First Citizen:\nYou are all resolved rather to die than to famish?"),
    ("Prompt 3 (Dialogue)", "MENENIUS:\nWhat work's, my countrymen, in hand?"),
    ("Prompt 4 (Dialogue)", "CORIOLANUS:\nWhat is the matter, you dissentious rogues?"),
    ("Prompt 5 (Completion)", "GLOUCESTER:\nNow is the winter of our discontent"),
]


def get_available_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_model(checkpoint_path: str, device: torch.device) -> LlmModel:
    """Instantiates LlmModel and loads checkpoint weights."""
    model_config = get_demo_model_config()
    model = LlmModel(model_config)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    # Handle PPO wrapper policy_model prefix if present
    clean_sd = {}
    for k, v in state_dict.items():
        if k.startswith("policy_model."):
            clean_sd[k[len("policy_model."):]] = v
        else:
            clean_sd[k] = v

    model.load_state_dict(clean_sd)
    model.to(device)
    model.eval()
    return model


def compute_validation_loss(model: LlmModel, device: torch.device, num_samples: int = 30) -> Tuple[float, float]:
    """Computes cross-entropy validation loss and perplexity on Shakespeare pretrain text."""
    data_path = os.path.join(CURRENT_DIR, "data", "pretrain.jsonl")
    if not os.path.exists(data_path):
        return 0.0, 0.0

    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line)["text"])
                if len(samples) >= num_samples:
                    break

    tokenizer = TrainerTools().tokenizer
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in samples:
            tokens = tokenizer.encode(text)
            if len(tokens) < 4:
                continue
            input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs.get("loss")
            if loss is None:
                logits = outputs["logits"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_tensor[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            total_loss += loss.item() * (len(tokens) - 1)
            total_tokens += (len(tokens) - 1)

    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


def evaluate_model_generations(
    model: LlmModel,
    device: torch.device,
    prompts: List[Tuple[str, str]],
    stage_name: str = "pretrain",
    max_new_tokens: int = 128
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Runs generation for each prompt.
    For SFT and later stages (DPO, GRPO, PPO), formats prompt as chat data format (<user>...</s><assistant>).
    """
    results = []
    total_reward = 0.0
    total_keyword_matches = 0
    tokenizer = TrainerTools().tokenizer
    is_chat_stage = stage_name.lower() in ("sft", "dpo", "grpo", "ppo")

    for label, raw_prompt in prompts:
        if is_chat_stage:
            eval_input = format_chat_prompt(raw_prompt, include_assistant_tag=True)
        else:
            eval_input = f"{raw_prompt}\nAll:" if "Citizen" in raw_prompt else raw_prompt

        gen_text = generate(
            model,
            prompt=eval_input,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            device=device
        )
        if gen_text is None:
            gen_text = ""

        # Separate prompt from completion
        if gen_text.startswith(eval_input):
            completion = gen_text[len(eval_input):].strip()
        else:
            completion = gen_text.strip()

        # Clean tags like <answer>, </answer>, </s> for display if present
        clean_comp = completion
        for tag in ["<answer>", "</answer>", "</s>", "<assistant>"]:
            clean_comp = clean_comp.replace(tag, "").strip()

        # Measure Shakespeare vocabulary presence in the generated completion
        words = [w.strip(".,;:?!'\"()[]").lower() for w in clean_comp.split() if w.strip()]
        kw_count = sum(1 for w in words if w in SHAKESPEARE_KEYWORDS)
        if kw_count > 0:
            total_keyword_matches += 1

        # Evaluate using demo reward function on completion
        prompt_ids = [torch.tensor(tokenizer.encode(eval_input), dtype=torch.long, device=device)]
        comp_ids = torch.tensor([tokenizer.encode(clean_comp if clean_comp else completion)], dtype=torch.long, device=device)
        rewards = demo_reward_func(prompt_ids, comp_ids, [None])
        score = rewards[0] if rewards else 0.0
        total_reward += score

        results.append({
            "label": label,
            "raw_prompt": raw_prompt,
            "eval_input": eval_input,
            "generation": completion,
            "clean_generation": clean_comp,
            "reward": score,
            "has_shakespeare_kw": kw_count > 0
        })

    mean_reward = total_reward / max(1, len(prompts))
    keyword_ratio = (total_keyword_matches / max(1, len(prompts))) * 100.0
    return results, keyword_ratio, mean_reward


def compare_all_checkpoints(device: torch.device):
    """Compares all existing checkpoints across the complete training pipeline."""
    ckpt_dir = os.path.join(CURRENT_DIR, "checkpoints")

    stages = [
        ("Base (Untrained)", os.path.join(ckpt_dir, "base_model.pt")),
        ("Pretrain", os.path.join(ckpt_dir, "pretrain_model.pt")),
        ("SFT", os.path.join(ckpt_dir, "sft_model.pt")),
        ("DPO", os.path.join(ckpt_dir, "dpo_model.pt")),
        ("GRPO", os.path.join(ckpt_dir, "grpo_model.pt")),
        ("PPO", os.path.join(ckpt_dir, "ppo_model.pt")),
    ]

    available_stages = [(name, path) for name, path in stages if os.path.exists(path)]

    if not available_stages:
        print("[Eval] No checkpoints found in demo/checkpoints/.")
        print("[Eval] Run 'python demo/prepare_data.py' or train stages first.")
        return

    print("=" * 90)
    print("                    SHAKESPEARE LLM STAGE-BY-STAGE EVALUATION REPORT")
    print("=" * 90)

    stage_summaries = []
    stage_generations: Dict[str, List[Dict[str, Any]]] = {}

    for name, path in available_stages:
        stage_key = name.split()[0].lower()
        print(f"\nEvaluating stage: [{name}] from {os.path.basename(path)} ...")
        model = load_checkpoint_model(path, device)
        val_loss, val_ppl = compute_validation_loss(model, device)
        gens, kw_ratio, mean_reward = evaluate_model_generations(
            model, device, BENCHMARK_PROMPTS, stage_name=stage_key
        )

        stage_summaries.append({
            "stage": name,
            "loss": val_loss,
            "ppl": val_ppl,
            "kw_ratio": kw_ratio,
            "reward": mean_reward,
        })
        stage_generations[name] = gens

    # Print summary metrics table
    print("\n" + "=" * 90)
    print("                           QUANTITATIVE METRICS SUMMARY")
    print("=" * 90)
    header = f"{'Stage':<18} | {'Val Loss':<10} | {'Perplexity':<12} | {'Shakespeare %':<14} | {'Mean Reward':<12}"
    print(header)
    print("-" * 90)
    for s in stage_summaries:
        print(f"{s['stage']:<18} | {s['loss']:<10.4f} | {s['ppl']:<12.2f} | {s['kw_ratio']:<13.1f}% | {s['reward']:<12.2f}")
    print("=" * 90)

    # Print qualitative prompt comparisons
    print("\n" + "=" * 90)
    print("                           QUALITATIVE TEXT GENERATION PROGRESSION")
    print("=" * 90)

    for i, (label, raw_prompt) in enumerate(BENCHMARK_PROMPTS):
        print(f"\n>>> [{label}]")
        print(f"Prompt Input: \"{raw_prompt}\"")
        print("-" * 90)
        for name, _ in available_stages:
            gen_item = stage_generations[name][i]
            gen_text = gen_item["clean_generation"] or gen_item["generation"]
            reward = gen_item["reward"]
            # Clean up display
            display_text = gen_text.replace("\n", " ").strip()
            if len(display_text) > 75:
                display_text = display_text[:72] + "..."
            print(f"  * {name:<18} (reward: {reward:+.2f}) -> \"{display_text}\"")

    print("\n" + "=" * 90)
    print("Evaluation complete.")


def evaluate_single_checkpoint(checkpoint_path: str, device: torch.device, stage_name: str = "pretrain", custom_prompt: Optional[str] = None):
    """Evaluates a single model checkpoint."""
    print(f"\n[Eval] Loading model from: {checkpoint_path}")
    model = load_checkpoint_model(checkpoint_path, device)

    val_loss, val_ppl = compute_validation_loss(model, device)
    print(f"[Metrics] Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

    prompts = [("Custom Prompt", custom_prompt)] if custom_prompt else BENCHMARK_PROMPTS
    gens, kw_ratio, mean_reward = evaluate_model_generations(model, device, prompts, stage_name=stage_name)

    print(f"[Metrics] Shakespeare Vocabulary Ratio: {kw_ratio:.1f}% | Mean Reward: {mean_reward:.2f}\n")
    print("Generated Responses:")
    for g in gens:
        print(f"\n>>> {g['label']}")
        print(f"Input : {g['raw_prompt']}")
        print(f"Output: {g['clean_generation'] or g['generation']}")
        print(f"Reward: {g['reward']:+.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Shakespeare LLM test checkpoints across stages.")
    parser.add_argument("--compare", action="store_true", help="Compare all available checkpoints side-by-side")
    parser.add_argument("--stage", type=str, default=None, choices=["base", "pretrain", "sft", "dpo", "grpo", "ppo"], help="Stage to evaluate")
    parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt string for evaluation")
    args = parser.parse_args()

    device = get_available_device()
    print(f"[Eval] Using device: {device}")

    if args.compare or (args.stage is None and args.checkpoint is None):
        compare_all_checkpoints(device)
    else:
        stage_name = args.stage or "pretrain"
        if args.checkpoint:
            ckpt_path = args.checkpoint
        else:
            ckpt_name = "base_model.pt" if stage_name == "base" else f"{stage_name}_model.pt"
            ckpt_path = os.path.join(CURRENT_DIR, "checkpoints", ckpt_name)

        evaluate_single_checkpoint(ckpt_path, device, stage_name=stage_name, custom_prompt=args.prompt)


if __name__ == "__main__":
    main()

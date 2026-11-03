import os
import sys
from typing import Optional, List, Dict, Tuple, Any, Union

# Ensure llm_model and llm_trainer can be imported from local repository
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))


def init_demo_env(stage: str = "default", zero_stage: Optional[int] = None):
    """
    Initialize all essential environment variables for llm_trainer and DeepSpeed.
    MUST be called before TrainerTools is instantiated.
    """
    token_dir = os.path.join(CURRENT_DIR, "data", "tokenizer")
    os.environ.setdefault("TOKEN_DIR", token_dir)

    stage_dir = f"{stage}_zero{zero_stage}" if zero_stage is not None else stage

    log_dir = os.path.join(CURRENT_DIR, "logs", stage_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.environ["LOG_DIR"] = log_dir

    ckpt_dir = os.path.join(CURRENT_DIR, "checkpoints", stage_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.environ["CHECKPOINT_DIR"] = ckpt_dir
    os.environ["DIST_CHECKPOINT_DIR"] = ckpt_dir
    os.environ.setdefault("CHECKPOINT_NAME", "ckpt.pth")
    os.environ.setdefault("EVAL_CHECKPOINT_NAME", "eval_ckpt.pth")

    for step_file in ("steps.pt", "steps.pth", "latest", "model.pth", "ckpt.pth"):
        p = os.path.join(ckpt_dir, step_file)
        if os.environ.get("CLEAN_STEPS", "1") == "1" and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    os.environ.setdefault("ENABLE_DCP", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # If running on CUDA with torchrun or ZeRO requested, enable DeepSpeed; otherwise fallback to none
    import torch as _torch
    is_distributed = ("RANK" in os.environ) or ("LOCAL_RANK" in os.environ) or ("WORLD_SIZE" in os.environ)
    if (is_distributed or (zero_stage in (2, 3))) and _torch.cuda.is_available():
        os.environ.setdefault("PARALLEL_TYPE", "ds")
    else:
        os.environ["PARALLEL_TYPE"] = "none"


import torch
from llm_model import ModelConfig, RoPEConfig
from llm_trainer import (
    TrainerTools,
    FileDataset,
    train_configs
)
from llm_trainer.train_configs import (
    TrainConfig,
    PretrainConfig,
    SFTConfig,
    DPOConfig,
    GRPOConfig,
    PPOConfig,
    OptimConfig,
    GenerateConfig,
    DataLoaderConfig,
    DsConfig,
    DsZero2Config,
    DsZero3Config,
    DsFp16Config,
    DsBf16Config,
)
from llm_trainer.generation_service import MultiTurnRLGenerationService


class ShakespeareFileDataset(FileDataset):
    """Concrete FileDataset returning file paths for Trainer data loading."""
    def __init__(self, file_paths: List[str]):
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> str:
        return self.file_paths[idx]


def get_demo_model_config() -> ModelConfig:
    """
    nanoGPT-inspired character-level Shakespeare Model (~0.61M parameters):
    - vocab_size=80 (from nanoGPT character tokenizer)
    - hidden_size=128, intermediate_size=256
    - 4 transformer layers, 4 attention heads (2 kv heads, head_dim=32)
    - max_position_embeddings=512 (expanded context window for character models)
    - tie_word_embeddings=False (safe for DeepSpeed ZeRO-3)
    """
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


def get_demo_ds_config(zero_stage: int = 2) -> DsConfig:
    """
    Builds DeepSpeed ZeRO-2 or ZeRO-3 configuration.
    Adaptively configures BF16 on Ampere+ GPUs or FP16 on Volta/Turing GPUs.
    """
    if zero_stage == 2:
        zero_config = DsZero2Config(
            allgather_partitions=True,
            reduce_scatter=True,
            overlap_comm=True,
            contiguous_gradients=True
        )
    elif zero_stage == 3:
        zero_config = DsZero3Config(
            stage3_gather_16bit_weights_on_model_save=True,
            overlap_comm=True,
            contiguous_gradients=True
        )
    else:
        raise ValueError(f"Unsupported zero_stage: {zero_stage}. Expected 2 or 3.")

    has_cuda = torch.cuda.is_available()
    cuda_bf16 = (
        has_cuda
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    cuda_fp16 = has_cuda and not cuda_bf16

    return DsConfig(
        zero_config=zero_config,
        fp16_config=DsFp16Config(enabled="auto" if cuda_fp16 else False, fp16_opt_level=None),
        bf16_config=DsBf16Config(enabled=cuda_bf16),
        wall_clock_breakdown=False
    )


def get_demo_optim_config(stage: str = "default") -> OptimConfig:
    """Configures optimizer with learning rate adapted to training stage."""
    stage_key = stage.lower()
    if stage_key in ("pretrain", "sft"):
        lr = 1e-3
        min_lr = 1e-5
    elif stage_key == "dpo":
        lr = 5e-5
        min_lr = 5e-6
    elif stage_key == "ppo":
        lr = 1e-4
        min_lr = 1e-5
    elif stage_key == "grpo":
        lr = 2e-4
        min_lr = 1e-5
    else:
        lr = 5e-4
        min_lr = 1e-5

    return OptimConfig(
        optim_type="adam",
        initial_lr=lr,
        auto_optimize_optimizer=True,
        enable_lr_scheduler=True,
        warmup_iters=2,
        min_lr=min_lr,
        max_lr=lr,
        cosine_annealing_period=50
    )


def demo_reward_func(
    prompt_ids: List[torch.Tensor],
    completion_ids: torch.Tensor,
    gt_answer_ids: List[Optional[torch.Tensor]],
    *,
    dones: Optional[List[bool]] = None,
    generation_masks: Optional[Union[List[List[bool]], torch.Tensor]] = None,
    feedbacks: Optional[List[str]] = None,
    **kwargs: Any
) -> List[float]:
    """
    Rule-based reward evaluator for Shakespeare line generation:
    - Evaluates ONLY model-generated tokens (filtering out multi-turn env feedback via generation_masks)
    - Base reward for meaningful output (+0.5 for >=15 chars, +0.8 for >=30 chars)
    - Strong negative penalty for empty / degenerate 1-token completions (-1.0)
    - Keyword rewards for Shakespearean vocabulary (up to +1.5)
    - Ground truth answer word overlap bonus (up to +1.0)
    - Completion bonus if task marked done AND output is non-trivial (+1.0)
    - Negative penalty for modern slang / refusal phrases (-1.0)
    """
    rewards = []
    tokenizer = TrainerTools().tokenizer
    pad_id = tokenizer.pad
    keywords = [
        "citizen", "speak", "hear", "people", "rome", "corn", "marcius", "caius",
        "resolved", "famish", "senate", "enemy", "lord", "king", "patricians",
        "shall", "thee", "thou", "verdict", "alive", "death", "die", "hath",
        "doth", "art", "well", "good", "sir", "all", "our", "men", "peace"
    ]
    penalties = ["whatever", "social media", "ai assistant", "error 404", "fast food", "video games"]
    special_tags = [
        "<answer>", "</answer>", "<think>", "</think>",
        "</s>", "<pad>", "<unk>", "<assistant>", "<user>", "<system>"
    ]

    batch_size = len(prompt_ids)
    for i in range(batch_size):
        comp_tokens = completion_ids[i]
        # In multi-turn RL, filter out environment feedback tokens using generation_masks
        if generation_masks is not None and i < len(generation_masks):
            mask_i = generation_masks[i]
            if isinstance(mask_i, torch.Tensor):
                mask_t = mask_i.to(device=comp_tokens.device, dtype=torch.bool)
            else:
                mask_t = torch.tensor(mask_i, dtype=torch.bool, device=comp_tokens.device)
            if mask_t.numel() == comp_tokens.numel():
                comp_tokens = comp_tokens[mask_t]

        if pad_id is not None:
            comp_tokens = comp_tokens[comp_tokens != pad_id]

        comp_text = tokenizer.decode(comp_tokens).lower() if comp_tokens.numel() > 0 else ""
        # Clean special tokens for text analysis
        text_clean = comp_text
        for tag in special_tags:
            text_clean = text_clean.replace(tag, "").strip()

        score = 0.0

        # Reward length: require meaningful content and strongly penalize empty/degenerate outputs
        if len(text_clean) >= 30:
            score += 0.8
        elif len(text_clean) >= 15:
            score += 0.5
        elif len(text_clean) >= 5:
            score -= 0.2
        else:
            score -= 1.0  # Strongly penalize empty / 1-char completions to prevent length collapse

        # Shakespeare keyword matching on clean text
        kw_hits = sum(1 for kw in keywords if kw in text_clean)
        if kw_hits > 0:
            score += min(1.5, 0.5 * kw_hits)

        # Dialogue format bonus (speaker colon, e.g. "ALL:", "CITIZEN:")
        if ":" in text_clean and len(text_clean) >= 15:
            score += 0.5

        for pen in penalties:
            if pen in text_clean:
                score -= 1.0
                break

        # Check ground truth answer match
        if gt_answer_ids is not None and i < len(gt_answer_ids) and gt_answer_ids[i] is not None:
            gt_text = tokenizer.decode(gt_answer_ids[i]).lower()
            gt_words = [w for w in gt_text.split() if len(w) > 3]
            match_count = sum(1 for w in gt_words if w in text_clean)
            if match_count > 0:
                score += min(1.0, 0.5 * match_count)

        # Only award completion bonus if the model actually generated meaningful content
        if dones is not None and i < len(dones) and dones[i] and len(text_clean) >= 15:
            score += 1.0

        rewards.append(float(score))

    return rewards


def demo_env_step(
    sample_idx: int,
    prompt_ids: List[int],
    history_tokens: List[int],
    generated_text: str
) -> Tuple[bool, str]:
    """
    Environment step callback for MultiTurnRLGenerationService.
    Simulates Roman citizen interaction:
    - If the model generates relevant non-empty dialogue, dialogue completes.
    - Otherwise returns citizen reaction feedback.
    """
    text_clean = generated_text.lower()
    for tag in ["<answer>", "</answer>", "<think>", "</think>", "</s>", "<pad>", "<unk>", "<assistant>"]:
        text_clean = text_clean.replace(tag, "").strip()

    is_done = len(text_clean) >= 15 and (
        "speak" in text_clean or "hear" in text_clean or len(history_tokens) > 16
    )
    feedback = "Citizen: Speak, speak! The assembly is listening."
    return is_done, feedback


def demo_format_feedback(feedback: str) -> str:
    """Formats environment feedback into a chat-template user turn."""
    return f"\n<user>{feedback}</s><assistant>"


def format_chat_prompt(user_content: str, include_assistant_tag: bool = True) -> str:
    """Formats single user text into standard chat template format."""
    tokenizer = TrainerTools().tokenizer
    conv = [{"role": "user", "content": user_content}]
    chat_str = tokenizer.apply_chat_template(conv, tokenizer=False)
    if include_assistant_tag:
        chat_str = f"{chat_str}{tokenizer.text_assistant}{tokenizer.text_answer_start}"
    return chat_str


def get_demo_eval_prompts(stage: str = "pretrain") -> List[str]:
    """
    Returns eval prompts tailored to training stage:
    - pretrain: plain text prompts suitable for causal LM continuation.
    - sft / dpo / grpo / ppo: chat formatted prompts using apply_chat_template (<user>...</s><assistant>).
    """
    raw_prompts = [
        "First Citizen:\nBefore we proceed any further, hear me speak.",
        "First Citizen:\nYou are all resolved rather to die than to famish?",
        "MENENIUS:\nWhat work's, my countrymen, in hand?",
        "CORIOLANUS:\nWhat is the matter, you dissentious rogues?",
        "GLOUCESTER:\nNow is the winter of our discontent",
    ]

    if stage.lower() in ("pretrain", "base"):
        return raw_prompts

    return [format_chat_prompt(p, include_assistant_tag=True) for p in raw_prompts]


def get_demo_multi_turn_service() -> MultiTurnRLGenerationService:
    """Builds MultiTurnRLGenerationService for multi-turn RL training."""
    return MultiTurnRLGenerationService(
        env_step=demo_env_step,
        format_feedback=demo_format_feedback,
        max_turns=2,
        max_consecutive_errors=2,
        env_timeout=15.0
    )


def get_base_checkpoint_path() -> str:
    """Returns absolute path to demo base model checkpoint."""
    path = os.path.join(CURRENT_DIR, "checkpoints", "base_model.pt")
    if not os.path.exists(path):
        import prepare_data
        prepare_data.save_base_checkpoint()
    return path


def get_upstream_weights_path(target_stage: str) -> str:
    """
    Returns the appropriate checkpoint path to initialize weights for target_stage:
    - sft: loads pretrain_model.pt (fallback base_model.pt)
    - dpo: loads sft_model.pt (fallback pretrain_model.pt -> base_model.pt)
    - grpo: loads sft_model.pt (fallback pretrain_model.pt -> base_model.pt)
    - ppo: loads sft_model.pt (fallback pretrain_model.pt -> base_model.pt)
    """
    ckpt_dir = os.path.join(CURRENT_DIR, "checkpoints")
    base_ckpt = get_base_checkpoint_path()
    pretrain_ckpt = os.path.join(ckpt_dir, "pretrain_model.pt")
    sft_ckpt = os.path.join(ckpt_dir, "sft_model.pt")

    if target_stage == "sft":
        if os.path.exists(pretrain_ckpt):
            print(f"[Weight Chaining] SFT loading pretrain weights from: {pretrain_ckpt}")
            return pretrain_ckpt
        print(f"[Weight Chaining Warning] {pretrain_ckpt} not found for SFT. Falling back to base_model.pt: {base_ckpt}")
        return base_ckpt

    if target_stage in ("dpo", "grpo", "ppo"):
        if os.path.exists(sft_ckpt):
            print(f"[Weight Chaining] {target_stage.upper()} loading SFT weights from: {sft_ckpt}")
            return sft_ckpt
        if os.path.exists(pretrain_ckpt):
            print(f"[Weight Chaining Warning] {sft_ckpt} not found for {target_stage.upper()}. Falling back to pretrain_model.pt: {pretrain_ckpt}")
            return pretrain_ckpt
        print(f"[Weight Chaining Warning] Neither SFT nor Pretrain checkpoint found. Falling back to base_model.pt: {base_ckpt}")
        return base_ckpt

    return base_ckpt


def save_stage_model(model: torch.nn.Module, stage_name: str) -> str:
    """
    Saves clean, gathered state_dict on rank 0 for ZeRO-2, ZeRO-3, or non-DS models.
    Handles PPO wrapper prefix unwrapping ('policy_model.').
    Synchronizes all ranks before continuing.
    """
    from llm_trainer.partition_utils import get_full_state_dict_on_rank0
    full_state_dict = get_full_state_dict_on_rank0(model)
    save_path = os.path.join(CURRENT_DIR, "checkpoints", f"{stage_name}_model.pt")

    if TrainerTools().parallel.is_main_process and full_state_dict is not None:
        # Check if wrapped (e.g. PPOTrainer PolicyAndValueModelWrapper)
        if any(k.startswith("policy_model.") for k in full_state_dict.keys()):
            clean_sd = {
                k[len("policy_model."):]: v
                for k, v in full_state_dict.items()
                if k.startswith("policy_model.")
            }
        else:
            clean_sd = {k: v for k, v in full_state_dict.items()}

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(clean_sd, save_path)
        print(f"\n[Checkpoint Saved] Successfully saved {stage_name} weights to: {save_path} ({len(clean_sd)} tensors)\n")

    TrainerTools().parallel.wait(f"save_{stage_name}")
    return save_path

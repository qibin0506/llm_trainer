import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

def main():
    parser = argparse.ArgumentParser(
        description="Unified Demo Test Runner covering Pretrain / SFT / DPO / GRPO / PPO with ZeRO-2/3, Chunked/Non-Chunked, and Single/Multi-Turn RL."
    )
    stage_default = os.environ.get("STAGE", "pretrain")
    zero_default = int(os.environ.get("ZERO", 2))
    chunked_default = os.environ.get("CHUNKED", "0") in ("1", "true", "True", "--chunked")
    rl_mode_default = os.environ.get("RL_MODE", "single")
    epochs_default = int(os.environ["EPOCHS"]) if os.environ.get("EPOCHS") else None
    batch_size_default = int(os.environ["BATCH_SIZE"]) if os.environ.get("BATCH_SIZE") else 8

    parser.add_argument(
        "--stage",
        type=str,
        default=stage_default,
        choices=["pretrain", "sft", "dpo", "grpo", "ppo"],
        help="Training stage to execute"
    )
    parser.add_argument(
        "--zero",
        type=int,
        default=zero_default,
        choices=[2, 3],
        help="DeepSpeed ZeRO optimization stage (2 or 3)"
    )
    parser.add_argument(
        "--chunked",
        action="store_true",
        default=chunked_default,
        help="Enable chunked cross entropy (for pretrain/sft) or chunked log_probs (for dpo/grpo/ppo)"
    )
    parser.add_argument(
        "--rl-mode",
        type=str,
        default=rl_mode_default,
        choices=["single", "multi"],
        help="RL rollout interaction mode: 'single' for single-turn or 'multi' for multi-turn (only for grpo/ppo)"
    )
    parser.add_argument("--epochs", type=int, default=epochs_default, help="Number of training epochs (stage default if not set)")
    parser.add_argument("--batch-size", type=int, default=batch_size_default, help="Batch size per GPU")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size (for grpo)")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size when chunked is enabled")
    parser.add_argument("--init-weights", type=str, default=os.environ.get("INIT_WEIGHTS"), help="Custom initial weights path")
    parser.add_argument("--ref-weights", type=str, default=os.environ.get("REF_WEIGHTS"), help="Custom reference model weights path")
    args = parser.parse_args()

    # Pass arguments to the respective runner
    sys.argv = [sys.argv[0]]
    sys.argv.extend(["--zero", str(args.zero)])
    if args.epochs is not None:
        sys.argv.extend(["--epochs", str(args.epochs)])
    sys.argv.extend(["--batch-size", str(args.batch_size)])
    sys.argv.extend(["--chunk-size", str(args.chunk_size)])
    if args.chunked:
        sys.argv.append("--chunked")
    if args.init_weights:
        sys.argv.extend(["--init-weights", args.init_weights])
    if args.ref_weights:
        sys.argv.extend(["--ref-weights", args.ref_weights])

    if args.stage == "pretrain":
        import run_pretrain
        run_pretrain.main()
    elif args.stage == "sft":
        import run_sft
        run_sft.main()
    elif args.stage == "dpo":
        import run_dpo
        run_dpo.main()
    elif args.stage == "grpo":
        sys.argv.extend(["--rl-mode", args.rl_mode, "--group-size", str(args.group_size)])
        import run_grpo
        run_grpo.main()
    elif args.stage == "ppo":
        sys.argv.extend(["--rl-mode", args.rl_mode])
        import run_ppo
        run_ppo.main()
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()

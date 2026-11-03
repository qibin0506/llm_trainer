import os
import argparse
from common import (
    init_demo_env,
    CURRENT_DIR,
    ShakespeareFileDataset,
    get_demo_model_config,
    get_demo_ds_config,
    get_demo_optim_config,
    get_upstream_weights_path,
    get_demo_eval_prompts,
    save_stage_model,
    demo_reward_func,
    get_demo_multi_turn_service,
)

def main():
    parser = argparse.ArgumentParser(description="Shakespeare GRPO Demo")
    parser.add_argument("--zero", type=int, default=2, choices=[2, 3], help="DeepSpeed ZeRO stage (2 or 3)")
    parser.add_argument("--chunked", action="store_true", help="Enable chunked log_probs")
    parser.add_argument("--rl-mode", type=str, default="single", choices=["single", "multi"], help="RL interaction mode (single-turn or multi-turn)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (default 2)")
    parser.add_argument("--batch-size", type=int, default=4, help="Rollout batch size per GPU")
    parser.add_argument("--group-size", type=int, default=4, help="GRPO group size (number of completions per prompt)")
    parser.add_argument("--grpo-batch-size", type=int, default=2, help="Micro-batch size for policy update")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size when chunked is enabled")
    parser.add_argument("--init-weights", type=str, default=None, help="Initial policy weights (defaults to sft_model.pt)")
    parser.add_argument("--ref-weights", type=str, default=None, help="Reference model weights (defaults to sft_model.pt)")
    args = parser.parse_args()

    stage_tag = f"grpo_{args.rl_mode}_chunked" if args.chunked else f"grpo_{args.rl_mode}"
    init_demo_env(stage=stage_tag, zero_stage=args.zero)

    from llm_trainer import GRPOTrainer
    from llm_trainer.train_configs import TrainConfig, GRPOConfig, DataLoaderConfig, GenerateConfig

    rl_data_path = os.path.join(CURRENT_DIR, "data", "rl.jsonl")
    dataset = ShakespeareFileDataset([rl_data_path])
    model_config = get_demo_model_config()
    ds_config = get_demo_ds_config(zero_stage=args.zero)
    optim_config = get_demo_optim_config("grpo")

    # Load upstream SFT weights for policy and reference models
    init_weights = args.init_weights or get_upstream_weights_path("grpo")
    ref_weights = args.ref_weights or init_weights

    generation_service = get_demo_multi_turn_service() if args.rl_mode == "multi" else None

    grpo_config = GRPOConfig(
        grpo_epochs=1,
        grpo_batch_size=args.grpo_batch_size,
        group_size=args.group_size,
        ref_model_weights_path=ref_weights,
        loss_beta=0.04,
        missing_eos_penalty=0.5,
        gradient_accumulation_steps=1,
        chunked_log_probs_size=args.chunk_size if args.chunked else None,
        generate_config=GenerateConfig(max_seq_len=256, top_p=1.0)
    )

    train_config = TrainConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        model_config=model_config,
        init_weights_path=init_weights,
        file_dataset=dataset,
        dataset_block_size=512,
        data_loader_config=DataLoaderConfig(num_workers=0, shuffle=False),
        optim_config=optim_config,
        ds_config=ds_config,
        grpo_config=grpo_config,
        eval_config=GenerateConfig(max_seq_len=256, temperature=0.8),
        save_interval=200,
        eval_interval=50,
    )

    trainer = GRPOTrainer(
        train_config=train_config,
        reward_func=demo_reward_func,
        generation_service=generation_service,
        eval_prompts=get_demo_eval_prompts("grpo")
    )

    print(f"=== Starting GRPO (ZeRO-{args.zero}, Mode={args.rl_mode}-turn, Chunked={args.chunked}, Epochs={args.epochs}) ===")
    print(f"=== Policy Init Weights: {init_weights} | Reference Weights: {ref_weights} ===")
    trainer.train()

    # Save full gathered weights
    save_path = save_stage_model(trainer.train_model, stage_tag)
    save_stage_model(trainer.train_model, "grpo")
    print(f"=== GRPO Completed Successfully. Weights saved to {save_path} ===")


if __name__ == "__main__":
    main()

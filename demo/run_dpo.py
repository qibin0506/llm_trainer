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
)

def main():
    parser = argparse.ArgumentParser(description="Shakespeare DPO Demo")
    parser.add_argument("--zero", type=int, default=2, choices=[2, 3], help="DeepSpeed ZeRO stage (2 or 3)")
    parser.add_argument("--chunked", action="store_true", help="Enable chunked log_probs")
    parser.add_argument("--loss-type", type=str, default="dpo", choices=["dpo", "simpo"], help="DPO loss type")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs (default 2)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU (real micro batch is batch_size * 2)")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size when chunked is enabled")
    parser.add_argument("--init-weights", type=str, default=None, help="Initial policy weights (defaults to sft_model.pt)")
    parser.add_argument("--ref-weights", type=str, default=None, help="Reference model weights (defaults to sft_model.pt)")
    args = parser.parse_args()

    stage_tag = "dpo_chunked" if args.chunked else "dpo"
    init_demo_env(stage=stage_tag, zero_stage=args.zero)

    from llm_trainer import DPOTrainer
    from llm_trainer.train_configs import TrainConfig, DPOConfig, DataLoaderConfig, GenerateConfig

    dpo_data_path = os.path.join(CURRENT_DIR, "data", "dpo.jsonl")
    dataset = ShakespeareFileDataset([dpo_data_path])
    model_config = get_demo_model_config()
    ds_config = get_demo_ds_config(zero_stage=args.zero)
    optim_config = get_demo_optim_config("dpo")

    # Load upstream SFT weights for policy and reference models
    init_weights = args.init_weights or get_upstream_weights_path("dpo")
    ref_weights = args.ref_weights or init_weights

    dpo_config = DPOConfig(
        ref_model_weights_path=ref_weights if args.loss_type == "dpo" else None,
        loss_type=args.loss_type,
        loss_beta=0.1,
        nll_loss_coef=0.1,
        mask_prompt=True,
        gradient_accumulation_steps=1,
        chunked_log_probs_size=args.chunk_size if args.chunked else None
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
        dpo_config=dpo_config,
        eval_config=GenerateConfig(max_seq_len=256, temperature=0.8),
        save_interval=200,
        eval_interval=50,
    )

    trainer = DPOTrainer(
        train_config=train_config,
        eval_prompts=get_demo_eval_prompts("dpo")
    )

    print(f"=== Starting DPO (ZeRO-{args.zero}, Loss={args.loss_type}, Chunked={args.chunked}, Epochs={args.epochs}) ===")
    print(f"=== Policy Init Weights: {init_weights} | Reference Weights: {ref_weights} ===")
    trainer.train()

    # Save full gathered weights to demo/checkpoints/
    save_path = save_stage_model(trainer.train_model, stage_tag)
    save_stage_model(trainer.train_model, "dpo")
    print(f"=== DPO Completed Successfully. Weights saved to {save_path} ===")


if __name__ == "__main__":
    main()

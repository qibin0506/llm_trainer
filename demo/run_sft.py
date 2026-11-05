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
    parser = argparse.ArgumentParser(description="Shakespeare SFT Demo")
    parser.add_argument("--zero", type=int, default=2, choices=[2, 3], help="DeepSpeed ZeRO stage (2 or 3)")
    parser.add_argument("--chunked", action="store_true", help="Enable chunked cross entropy")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default 3)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size when chunked is enabled")
    parser.add_argument("--init-weights", type=str, default=None, help="Initial weights path (defaults to pretrain_model.pt)")
    args = parser.parse_args()

    stage_tag = "sft_chunked" if args.chunked else "sft"
    init_demo_env(stage=stage_tag, zero_stage=args.zero)

    from llm_trainer import SFTTrainer
    from llm_trainer.train_configs import TrainConfig, SFTConfig, DataLoaderConfig, GenerateConfig

    sft_data_path = os.path.join(CURRENT_DIR, "data", "sft.jsonl")
    dataset = ShakespeareFileDataset([sft_data_path])
    model_config = get_demo_model_config()
    ds_config = get_demo_ds_config(zero_stage=args.zero)
    optim_config = get_demo_optim_config("sft")

    # Load upstream weights (Pretrain weights chained to SFT)
    init_weights = args.init_weights or get_upstream_weights_path("sft")

    sft_config = SFTConfig(
        dataset_block_size=512,
        mask_prompt=True,
        gradient_accumulation_steps=1,
        chunked_cross_entropy_size=args.chunk_size if args.chunked else None
    )

    train_config = TrainConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        model_config=model_config,
        init_weights_path=init_weights,
        file_dataset=dataset,
        data_loader_config=DataLoaderConfig(num_workers=0, shuffle=False),
        optim_config=optim_config,
        ds_config=ds_config,
        sft_config=sft_config,
        eval_config=GenerateConfig(max_new_tokens=256, temperature=0.8),
        save_interval=200,
        eval_interval=50,
    )

    trainer = SFTTrainer(
        train_config=train_config,
        eval_prompts=get_demo_eval_prompts("sft")
    )

    print(f"=== Starting SFT (ZeRO-{args.zero}, Chunked={args.chunked}, Epochs={args.epochs}) ===")
    print(f"=== Upstream Weights: {init_weights} ===")
    trainer.train()

    # Save full gathered weights to demo/checkpoints/
    save_path = save_stage_model(trainer.train_model, stage_tag)
    save_stage_model(trainer.train_model, "sft")
    print(f"=== SFT Completed Successfully. Weights saved to {save_path} ===")


if __name__ == "__main__":
    main()

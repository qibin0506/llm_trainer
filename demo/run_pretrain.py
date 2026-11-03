import os
import argparse
from common import (
    init_demo_env,
    CURRENT_DIR,
    ShakespeareFileDataset,
    get_demo_model_config,
    get_demo_ds_config,
    get_demo_optim_config,
    get_demo_eval_prompts,
    save_stage_model,
)

def main():
    parser = argparse.ArgumentParser(description="Shakespeare Pretrain Demo")
    parser.add_argument("--zero", type=int, default=2, choices=[2, 3], help="DeepSpeed ZeRO stage (2 or 3)")
    parser.add_argument("--chunked", action="store_true", help="Enable chunked cross entropy")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default 3)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--chunk-size", type=int, default=32, help="Chunk size when chunked is enabled")
    parser.add_argument("--init-weights", type=str, default=None, help="Optional initial weights path")
    args = parser.parse_args()

    stage_tag = "pretrain_chunked" if args.chunked else "pretrain"
    init_demo_env(stage=stage_tag, zero_stage=args.zero)

    from llm_trainer import Trainer
    from llm_trainer.train_configs import TrainConfig, PretrainConfig, DataLoaderConfig, GenerateConfig

    pretrain_data_path = os.path.join(CURRENT_DIR, "data", "pretrain.jsonl")
    dataset = ShakespeareFileDataset([pretrain_data_path])
    model_config = get_demo_model_config()
    ds_config = get_demo_ds_config(zero_stage=args.zero)
    optim_config = get_demo_optim_config("pretrain")

    pretrain_config = PretrainConfig(
        gradient_accumulation_steps=1,
        chunked_cross_entropy_size=args.chunk_size if args.chunked else None
    )

    train_config = TrainConfig(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        model_config=model_config,
        init_weights_path=args.init_weights,
        file_dataset=dataset,
        dataset_block_size=512,
        data_loader_config=DataLoaderConfig(num_workers=0, shuffle=False),
        optim_config=optim_config,
        ds_config=ds_config,
        pretrain_config=pretrain_config,
        eval_config=GenerateConfig(max_seq_len=256, temperature=0.8),
        save_interval=200,
        eval_interval=50,
    )

    trainer = Trainer(
        train_config=train_config,
        eval_prompts=get_demo_eval_prompts("pretrain")
    )

    print(f"=== Starting Pretrain (ZeRO-{args.zero}, Chunked={args.chunked}, Epochs={args.epochs}) ===")
    trainer.train()

    # Save full gathered weights to demo/checkpoints/
    save_path = save_stage_model(trainer.train_model, stage_tag)
    save_stage_model(trainer.train_model, "pretrain")
    print(f"=== Pretrain Completed Successfully. Weights saved to {save_path} ===")


if __name__ == "__main__":
    main()

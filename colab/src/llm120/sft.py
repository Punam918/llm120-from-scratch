from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from llm120.posttrain_common import (
    evaluation_subset,
    finish_stage,
    linear_warmup_steps,
    load_split,
    load_tokenizer,
    require_cuda,
    resolve_model,
    resolve_resume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised instruction-tune the pretrained model")
    parser.add_argument("--model", type=Path, default=Path("checkpoints/llm120-english"))
    parser.add_argument("--data", type=Path, default=Path("data/posttrain/normalized"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/llm120-sft"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", default="auto", help="auto, none, or an explicit Trainer checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()
    model_path = resolve_model(args.model)
    tokenizer = load_tokenizer(model_path)
    train = load_split(args.data, "sft", "train")
    validation = evaluation_subset(load_split(args.data, "sft", "validation"))
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    config = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=linear_warmup_steps(
            rows=len(train),
            epochs=args.epochs,
            batch_size=2,
            accumulation_steps=8,
            max_steps=args.max_steps,
        ),
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_length=2048,
        assistant_only_loss=True,
        packing=False,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=10,
        logging_first_step=True,
        save_steps=500,
        save_total_limit=3,
        report_to="tensorboard",
        dataset_num_proc=12,
        dataloader_num_workers=4,
        seed=42,
        data_seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train,
        eval_dataset=validation,
        processing_class=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=resolve_resume(args.output, args.resume))
    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    final = finish_stage(trainer, tokenizer, args.output, metrics)
    print(f"SFT complete: {final}")


if __name__ == "__main__":
    main()

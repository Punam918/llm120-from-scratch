from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification
from trl import RewardConfig, RewardTrainer

from llm120.posttrain_common import (
    finish_stage,
    linear_warmup_steps,
    load_split,
    load_tokenizer,
    require_cuda,
    resolve_model,
    resolve_resume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HelpSteer2 human-preference reward model")
    parser.add_argument("--model", type=Path, default=Path("checkpoints/llm120-sft"))
    parser.add_argument("--data", type=Path, default=Path("data/posttrain/normalized"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/llm120-reward"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()
    model_path = resolve_model(args.model)
    tokenizer = load_tokenizer(model_path)
    train = load_split(args.data, "reward", "train")
    validation = load_split(args.data, "reward", "validation")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        dtype=torch.float32,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    config = RewardConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        max_length=1024,
        center_rewards_coefficient=0.01,
        eval_strategy="steps",
        eval_steps=250,
        logging_steps=10,
        logging_first_step=True,
        save_steps=250,
        save_total_limit=3,
        report_to="tensorboard",
        dataset_num_proc=12,
        dataloader_num_workers=4,
        seed=42,
        data_seed=42,
    )
    trainer = RewardTrainer(
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
    accuracy = metrics.get("eval_accuracy", float("nan"))
    print(f"Reward model complete: {final} (held-out preference accuracy {accuracy:.3f})")


if __name__ == "__main__":
    main()

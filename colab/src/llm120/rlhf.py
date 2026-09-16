from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RLOOConfig, RLOOTrainer

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
    parser = argparse.ArgumentParser(description="Run conservative reward-model-based RLOO RLHF")
    parser.add_argument("--model", type=Path, default=Path("checkpoints/llm120-dpo"))
    parser.add_argument("--reward-model", type=Path, default=Path("checkpoints/llm120-reward"))
    parser.add_argument("--data", type=Path, default=Path("data/posttrain/normalized"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/llm120-assistant"))
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--resume", default="auto")
    parser.add_argument(
        "--allow-weak-reward-model",
        action="store_true",
        help="Bypass the held-out preference-accuracy safety gate (not recommended)",
    )
    return parser.parse_args()


def reward_accuracy(reward_path: Path) -> float:
    metrics_path = reward_path / "stage-metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Reward model metrics not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics.get("eval_accuracy", "nan"))


def completion_health_reward(
    completions: list[Any], completion_ids: list[list[int]], **_: Any
) -> list[float]:
    """Small anti-degeneration signal; the learned human reward remains dominant."""
    scores: list[float] = []
    for completion, token_ids in zip(completions, completion_ids, strict=True):
        if isinstance(completion, list):
            text = " ".join(str(item.get("content", "")) for item in completion)
        else:
            text = str(completion)
        words = text.lower().split()
        trigrams = [tuple(words[index : index + 3]) for index in range(max(0, len(words) - 2))]
        repeated = len(trigrams) - len(set(trigrams))
        repetition_penalty = repeated / max(1, len(trigrams))
        empty_penalty = 1.0 if len(words) < 2 else 0.0
        scores.append(-(min(1.0, 5.0 * repetition_penalty) + empty_penalty))
    return scores


def main() -> None:
    args = parse_args()
    require_cuda()
    policy_path = resolve_model(args.model)
    reward_path = resolve_model(args.reward_model)
    accuracy = reward_accuracy(reward_path)
    if (not torch.isfinite(torch.tensor(accuracy)) or accuracy < 0.55) and not args.allow_weak_reward_model:
        raise RuntimeError(
            f"Refusing RLHF: held-out reward-model accuracy is {accuracy:.3f}; require >= 0.55. "
            "Improve the reward model or explicitly pass --allow-weak-reward-model."
        )

    tokenizer = load_tokenizer(policy_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_path, use_fast=True)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_path, dtype=torch.float32, attn_implementation="sdpa"
    )
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    train = load_split(args.data, "rlhf", "train")
    validation = evaluation_subset(load_split(args.data, "rlhf", "validation"), maximum=128)
    config = RLOOConfig(
        output_dir=str(args.output),
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        warmup_steps=linear_warmup_steps(
            rows=len(train),
            epochs=1.0,
            batch_size=2,
            accumulation_steps=8,
            max_steps=args.max_steps,
        ),
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        model_init_kwargs={"dtype": torch.float32, "attn_implementation": "sdpa"},
        num_generations=2,
        max_completion_length=128,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.05,
        beta=0.1,
        num_iterations=1,
        epsilon=0.2,
        reward_weights=[1.0, 0.1],
        normalize_advantages=True,
        mask_truncated_completions=True,
        eval_strategy="steps",
        eval_steps=50,
        logging_steps=5,
        logging_first_step=True,
        log_completions=True,
        num_completions_to_print=4,
        save_steps=50,
        save_total_limit=3,
        report_to="tensorboard",
        dataloader_num_workers=2,
        seed=42,
        data_seed=42,
    )
    trainer = RLOOTrainer(
        model=str(policy_path),
        reward_funcs=[reward_model, completion_health_reward],
        args=config,
        train_dataset=train,
        eval_dataset=validation,
        processing_class=tokenizer,
        reward_processing_classes=[reward_tokenizer, None],
    )
    result = trainer.train(resume_from_checkpoint=resolve_resume(args.output, args.resume))
    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    metrics["reward_model_eval_accuracy"] = accuracy
    final = finish_stage(trainer, tokenizer, args.output, metrics)
    print(f"RLHF complete: {final}")


if __name__ == "__main__":
    main()

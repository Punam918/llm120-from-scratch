from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Post-training requires CUDA. Run llm120-doctor first.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The configured post-training recipe requires BF16 support.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def resolve_model(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = [path / "final", path]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return candidate
    checkpoint_dirs = sorted(path.glob("checkpoint-*"))
    for checkpoint in reversed(checkpoint_dirs):
        nested = checkpoint / "model"
        if (nested / "config.json").exists():
            return nested
        if (checkpoint / "config.json").exists():
            return checkpoint
    raise FileNotFoundError(f"No saved Transformers model found at or below {path}")


def load_tokenizer(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no padding token")
    if not tokenizer.chat_template:
        raise ValueError("Tokenizer has no chat template; retrain it with llm120-tokenizer")
    return tokenizer


def load_split(data_dir: Path, name: str, split: str) -> Dataset:
    path = data_dir / f"{name}-{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing normalized dataset: {path}. Run llm120-prepare-posttrain.")
    return load_dataset("parquet", data_files={split: str(path)})[split]


def evaluation_subset(dataset: Dataset, maximum: int = 1_000) -> Dataset:
    return dataset if len(dataset) <= maximum else dataset.select(range(maximum))


def linear_warmup_steps(
    *, rows: int, epochs: float, batch_size: int, accumulation_steps: int, max_steps: int
) -> int:
    """Resolve a 10% warmup without relying on Transformers' deprecated warmup_ratio."""
    if max_steps > 0:
        total_steps = max_steps
    else:
        batches_per_epoch = math.ceil(rows / batch_size)
        total_steps = math.ceil(epochs * batches_per_epoch / accumulation_steps)
    return 0 if total_steps < 10 else max(1, int(total_steps * 0.1))


def resolve_resume(output: Path, value: str) -> str | None:
    last = get_last_checkpoint(str(output)) if output.exists() else None
    if value == "auto":
        return last
    if value == "none":
        if last is not None:
            raise RuntimeError(f"{output} contains checkpoints; use --resume auto or a new output path")
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return str(path)


def finish_stage(trainer: Any, tokenizer: Any, output: Path, metrics: dict[str, Any]) -> Path:
    final = output / "final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    serializable = {
        key: float(value) if isinstance(value, (float, int)) else str(value)
        for key, value in metrics.items()
    }
    (final / "stage-metrics.json").write_text(json.dumps(serializable, indent=2) + "\n")
    return final

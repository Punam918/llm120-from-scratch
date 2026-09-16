from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def checkpoints(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    found = []
    if root.exists():
        for path in root.iterdir():
            match = CHECKPOINT_PATTERN.match(path.name)
            if path.is_dir() and match:
                found.append((int(match.group(1)), path))
    return [path for _, path in sorted(found)]


def latest_checkpoint(output_dir: str | Path) -> Path | None:
    found = checkpoints(output_dir)
    return found[-1] if found else None


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    output_dir: str | Path,
    step: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    metadata: dict,
    *,
    keep_last: int,
) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root / f"checkpoint-{step:09d}"
    temporary = root / f".checkpoint-{step:09d}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model
    unwrapped.save_pretrained(temporary / "model", safe_serialization=True)
    tokenizer.save_pretrained(temporary / "model")
    state = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(),
        "metadata": metadata,
    }
    torch.save(state, temporary / "trainer-state.pt")
    (temporary / "metadata.json").write_text(
        json.dumps(metadata | {"step": step}, indent=2) + "\n", encoding="utf-8"
    )
    temporary.rename(final)
    old = checkpoints(root)[:-keep_last]
    for path in old:
        shutil.rmtree(path)
    return final


def load_trainer_state(
    checkpoint: str | Path,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> dict:
    state = torch.load(
        Path(checkpoint) / "trainer-state.pt", map_location="cpu", weights_only=False
    )
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    restore_rng_state(state["rng"])
    return state

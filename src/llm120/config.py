from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    output_dir: Path
    seed: int


@dataclass(frozen=True)
class PathsConfig:
    tokenizer: Path
    train_manifest: Path
    validation_manifest: Path


@dataclass(frozen=True)
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    initializer_range: float
    tie_word_embeddings: bool


@dataclass(frozen=True)
class TrainingConfig:
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    max_tokens: int
    learning_rate: float
    warmup_steps: int
    decay_ratio: float
    min_learning_rate_ratio: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    max_grad_norm: float
    precision: str
    gradient_checkpointing: bool
    torch_compile: bool
    allow_tf32: bool
    fused_optimizer: bool
    log_every_steps: int
    eval_every_steps: int
    eval_batches: int
    save_every_steps: int
    keep_last_checkpoints: int

    @property
    def tokens_per_step(self) -> int:
        return self.sequence_length * self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def max_steps(self) -> int:
        return math.ceil(self.max_tokens / self.tokens_per_step)


@dataclass(frozen=True)
class Config:
    project: ProjectConfig
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig
    source_path: Path


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path) -> Config:
    source = Path(path).expanduser().resolve()
    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = source.parent.parent
    project = dict(raw["project"])
    paths = dict(raw["paths"])
    project["output_dir"] = _resolve(root, project["output_dir"])
    for key in ("tokenizer", "train_manifest", "validation_manifest"):
        paths[key] = _resolve(root, paths[key])
    config = Config(
        project=ProjectConfig(**project),
        paths=PathsConfig(**paths),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        source_path=source,
    )
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    model = config.model
    train = config.training
    if model.hidden_size % model.num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if model.num_attention_heads % model.num_key_value_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    if train.sequence_length > model.max_position_embeddings:
        raise ValueError("sequence_length exceeds max_position_embeddings")
    if train.precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be bf16, fp16, or fp32")
    if not 0.0 < train.decay_ratio < 1.0:
        raise ValueError("decay_ratio must be between zero and one")
    if train.warmup_steps >= train.max_steps:
        raise ValueError("warmup_steps must be lower than max_steps")

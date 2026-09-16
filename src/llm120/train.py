from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, LlamaForCausalLM

from llm120.checkpoint import latest_checkpoint, load_trainer_state, save_checkpoint
from llm120.config import Config, load_config
from llm120.data import TokenBlockCorpus
from llm120.model import make_model, optimizer_groups, parameter_counts
from llm120.schedule import wsd_factor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the 125M model on one GPU")
    parser.add_argument("--config", type=Path, default=Path("configs/train_125m.yaml"))
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto, none, or an explicit checkpoint directory",
    )
    parser.add_argument("--max-steps", type=int, help="Override max steps (useful for smoke tests)")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def precision_context(precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    corpus: TokenBlockCorpus,
    batches: int,
    batch_size: int,
    precision: str,
) -> tuple[float, float]:
    model.eval()
    losses = []
    for index in range(min(batches, math.ceil(corpus.num_blocks / batch_size))):
        input_ids = corpus.batch(index * batch_size, batch_size).pin_memory().to(
            "cuda", non_blocking=True
        )
        with precision_context(precision):
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        losses.append(float(output.loss.detach()))
    model.train()
    loss = sum(losses) / len(losses)
    return loss, math.exp(min(loss, 20.0))


def resolve_resume(value: str, output_dir: Path) -> Path | None:
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        return latest_checkpoint(output_dir)
    path = Path(value).expanduser().resolve()
    if not (path / "trainer-state.pt").exists():
        raise FileNotFoundError(f"Not a training checkpoint: {path}")
    return path


def make_optimizer(model: LlamaForCausalLM, config: Config) -> torch.optim.AdamW:
    train = config.training
    kwargs = dict(
        params=optimizer_groups(model, train.weight_decay),
        lr=train.learning_rate,
        betas=(train.adam_beta1, train.adam_beta2),
        eps=train.adam_epsilon,
    )
    if train.fused_optimizer:
        try:
            return torch.optim.AdamW(**kwargs, fused=True)
        except (RuntimeError, TypeError):
            print("Fused AdamW is unavailable; falling back to standard AdamW", flush=True)
    return torch.optim.AdamW(**kwargs)


def run(config: Config, resume_value: str, max_steps_override: int | None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run llm120-doctor before training.")
    train = config.training
    if train.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU/PyTorch build does not support BF16")
    torch.backends.cuda.matmul.allow_tf32 = train.allow_tf32
    torch.backends.cudnn.allow_tf32 = train.allow_tf32
    torch.set_float32_matmul_precision("high")
    seed_everything(config.project.seed)

    tokenizer = AutoTokenizer.from_pretrained(config.paths.tokenizer, use_fast=True)
    train_corpus = TokenBlockCorpus(
        config.paths.train_manifest,
        train.sequence_length,
        seed=config.project.seed,
        shuffle=True,
    )
    validation_corpus = TokenBlockCorpus(
        config.paths.validation_manifest,
        train.sequence_length,
        seed=config.project.seed,
        shuffle=False,
    )
    output_dir = config.project.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resume = resolve_resume(resume_value, output_dir)
    if resume is None and latest_checkpoint(output_dir) is not None:
        raise RuntimeError(
            f"{output_dir} already contains checkpoints. Use --resume auto or choose a new output_dir."
        )
    if resume:
        print(f"Resuming model from {resume}", flush=True)
        model = LlamaForCausalLM.from_pretrained(
            resume / "model", dtype=torch.float32, attn_implementation="sdpa"
        )
    else:
        model = make_model(config.model, tokenizer)
        model.config._attn_implementation = "sdpa"
    model.config.use_cache = False
    if train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda")
    total_parameters, trainable_parameters = parameter_counts(model)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=train.precision == "fp16")

    config_hash = file_sha256(config.source_path)
    run_identity = {
        "config_sha256": config_hash,
        "tokenizer_sha256": file_sha256(config.paths.tokenizer / "tokenizer.json"),
        "train_manifest_sha256": file_sha256(config.paths.train_manifest),
        "validation_manifest_sha256": file_sha256(config.paths.validation_manifest),
        "parameters": total_parameters,
        "tokens_per_step": train.tokens_per_step,
    }
    start_step = 0
    if resume:
        state = load_trainer_state(resume, optimizer, scaler)
        previous = state.get("metadata", {})
        for key in ("config_sha256", "tokenizer_sha256", "train_manifest_sha256"):
            if previous.get(key) != run_identity[key]:
                raise RuntimeError(f"Resume refused: {key} differs from the checkpoint")
        start_step = int(state["step"])

    max_steps = max_steps_override or train.max_steps
    if max_steps <= start_step:
        print(f"Nothing to do: checkpoint step {start_step:,} >= target {max_steps:,}")
        return
    if train.torch_compile:
        print("Compiling model; the first step can take several minutes", flush=True)
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    run_info = run_identity | {
        "name": config.project.name,
        "max_steps": max_steps,
        "max_tokens": max_steps * train.tokens_per_step,
        "train_usable_tokens": train_corpus.usable_tokens,
        "validation_usable_tokens": validation_corpus.usable_tokens,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    (output_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")
    log_path = output_dir / "metrics.jsonl"
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        print(f"Received signal {signum}; checkpointing after this optimizer step", flush=True)
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        f"{total_parameters:,} parameters ({trainable_parameters:,} trainable) | "
        f"{train.tokens_per_step:,} tokens/step | steps {start_step:,}..{max_steps:,}",
        flush=True,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    interval_start = time.perf_counter()
    interval_loss = 0.0
    interval_steps = 0
    last_step = start_step
    for step in range(start_step, max_steps):
        step_loss = 0.0
        first_sample = step * train.gradient_accumulation_steps * train.micro_batch_size
        for accumulation in range(train.gradient_accumulation_steps):
            sample = first_sample + accumulation * train.micro_batch_size
            input_ids = train_corpus.batch(sample, train.micro_batch_size).pin_memory().to(
                "cuda", non_blocking=True
            )
            with precision_context(train.precision):
                output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                loss = output.loss / train.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at optimizer step {step}: {loss.item()}")
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train.max_grad_norm)
        factor = wsd_factor(
            step,
            max_steps=max_steps,
            warmup_steps=train.warmup_steps,
            decay_ratio=train.decay_ratio,
            min_ratio=train.min_learning_rate_ratio,
        )
        learning_rate = train.learning_rate * factor
        set_learning_rate(optimizer, learning_rate)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        completed_step = step + 1
        last_step = completed_step
        interval_loss += step_loss
        interval_steps += 1

        if completed_step % train.log_every_steps == 0 or completed_step == 1:
            elapsed = time.perf_counter() - interval_start
            mean_loss = interval_loss / interval_steps
            tokens_second = interval_steps * train.tokens_per_step / elapsed
            metrics = {
                "step": completed_step,
                "tokens": completed_step * train.tokens_per_step,
                "loss": mean_loss,
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "tokens_per_second": tokens_second,
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            for key, value in metrics.items():
                if key not in {"step", "tokens"}:
                    writer.add_scalar(f"train/{key}", value, completed_step)
            print(
                f"step {completed_step:,}/{max_steps:,} | loss {mean_loss:.4f} | "
                f"lr {learning_rate:.3e} | {tokens_second:,.0f} tok/s | "
                f"VRAM {metrics['peak_vram_gib']:.2f} GiB",
                flush=True,
            )
            interval_start = time.perf_counter()
            interval_loss = 0.0
            interval_steps = 0
            torch.cuda.reset_peak_memory_stats()

        if completed_step % train.eval_every_steps == 0:
            validation_loss, perplexity = evaluate(
                model,
                validation_corpus,
                train.eval_batches,
                train.micro_batch_size,
                train.precision,
            )
            writer.add_scalar("validation/loss", validation_loss, completed_step)
            writer.add_scalar("validation/perplexity", perplexity, completed_step)
            print(
                f"validation | step {completed_step:,} | loss {validation_loss:.4f} | "
                f"perplexity {perplexity:.2f}",
                flush=True,
            )

        should_save = completed_step % train.save_every_steps == 0 or stop_requested
        if should_save:
            checkpoint = save_checkpoint(
                output_dir,
                completed_step,
                model,
                tokenizer,
                optimizer,
                scaler,
                run_identity | {"tokens": completed_step * train.tokens_per_step},
                keep_last=train.keep_last_checkpoints,
            )
            print(f"Saved {checkpoint}", flush=True)
        if stop_requested:
            break

    if not stop_requested and last_step == max_steps:
        checkpoint = latest_checkpoint(output_dir)
        if checkpoint is None or checkpoint.name != f"checkpoint-{last_step:09d}":
            checkpoint = save_checkpoint(
                output_dir,
                last_step,
                model,
                tokenizer,
                optimizer,
                scaler,
                run_identity | {"tokens": last_step * train.tokens_per_step, "final": True},
                keep_last=train.keep_last_checkpoints,
            )
        print(f"Training complete: {checkpoint}", flush=True)
    writer.close()


def main() -> None:
    args = parse_args()
    run(load_config(args.config), args.resume, args.max_steps)


if __name__ == "__main__":
    main()

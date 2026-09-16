from __future__ import annotations

import argparse
import contextlib
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from llm120.config import load_config
from llm120.model import make_model, parameter_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure realistic forward/backward throughput and VRAM")
    parser.add_argument("--config", type=Path, default=Path("configs/train_125m.yaml"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Use a synthetic tokenizer of this size (allows benchmarking before data setup)",
    )
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Benchmark torch.compile with the same mode supported by the trainer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    config = load_config(args.config)
    train = config.training
    torch.backends.cuda.matmul.allow_tf32 = train.allow_tf32
    torch.backends.cudnn.allow_tf32 = train.allow_tf32
    torch.set_float32_matmul_precision("high")
    sequence_length = args.sequence_length or train.sequence_length
    micro_batch_size = args.micro_batch_size or train.micro_batch_size
    if args.vocab_size:
        tokenizer = type(
            "SyntheticTokenizer",
            (),
            {
                "__len__": lambda self: args.vocab_size,
                "bos_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": 2,
            },
        )()
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.paths.tokenizer, use_fast=True)
    model = make_model(config.model, tokenizer)
    model.config._attn_implementation = "sdpa"
    model.config.use_cache = False
    if train.gradient_checkpointing and not args.no_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda").train()
    if args.compile or train.torch_compile:
        print("Compiling model; the first iteration can take several minutes", flush=True)
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    inputs = torch.randint(
        0, len(tokenizer), (micro_batch_size, sequence_length), device="cuda"
    )
    dtype = torch.bfloat16 if train.precision == "bf16" else torch.float16
    total, _ = parameter_counts(model)
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for step in range(args.steps + 2):
        torch.cuda.synchronize()
        started = time.perf_counter()
        context = (
            torch.autocast("cuda", dtype=dtype)
            if train.precision != "fp32"
            else contextlib.nullcontext()
        )
        with context:
            loss = model(input_ids=inputs, labels=inputs, use_cache=False).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step >= 2:
            durations.append(time.perf_counter() - started)
    tokens_per_second = sequence_length * micro_batch_size / (sum(durations) / len(durations))
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Parameters: {total:,}")
    print(f"Sequence: {sequence_length:,}; micro batch: {micro_batch_size}")
    print(f"Throughput: {tokens_per_second:,.0f} tokens/second")
    print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"Estimated 10B-token wall time at 90% utilization: {10e9 / tokens_per_second / 0.9 / 86400:.1f} days")


if __name__ == "__main__":
    main()

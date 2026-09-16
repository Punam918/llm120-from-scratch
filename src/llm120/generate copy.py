from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm120.checkpoint import latest_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a base-model continuation")
    parser.add_argument("prompt", nargs="?", default="The solar system consists of")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/llm120-english"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or latest_checkpoint(args.output_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found below {args.output_dir}")
    model_path = checkpoint / "model" if (checkpoint / "model").exists() else checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids
    if tokenizer.bos_token_id is not None:
        bos = torch.full((1, 1), tokenizer.bos_token_id, dtype=input_ids.dtype)
        input_ids = torch.cat((bos, input_ids), dim=1)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    kwargs = {
        "input_ids": input_ids.to(device),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if args.sample:
        kwargs.update(temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        output = model.generate(**kwargs)
    print(tokenizer.decode(output[0][1:], skip_special_tokens=True))


if __name__ == "__main__":
    main()

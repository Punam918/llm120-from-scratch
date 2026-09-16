from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from llm120.posttrain_common import load_tokenizer, resolve_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the instruction-tuned assistant")
    parser.add_argument("prompt", nargs="?", default="Explain why the sky is blue in simple terms.")
    parser.add_argument("--model", type=Path, default=Path("checkpoints/llm120-assistant"))
    parser.add_argument(
        "--system", default="You are a helpful, honest, and concise AI assistant."
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model(args.model)
    tokenizer = load_tokenizer(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    encoded = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generate_kwargs = {
        **encoded,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": not args.greedy,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.05,
    }
    if not args.greedy:
        generate_kwargs.update(temperature=args.temperature, top_p=args.top_p)
    with torch.inference_mode():
        output = model.generate(**generate_kwargs)
    completion = output[0, encoded["input_ids"].shape[1] :]
    print(tokenizer.decode(completion, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()

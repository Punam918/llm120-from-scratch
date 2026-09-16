from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

from llm120.config import load_config
from llm120.data import load_manifest


def validate_manifest(path: Path, tokenizer_hash: str) -> tuple[bool, str]:
    try:
        manifest = load_manifest(path)
        if manifest.get("dtype") != "uint16" or manifest.get("byte_order") != "little":
            raise ValueError("expected little-endian uint16")
        if manifest.get("tokenizer_sha256") != tokenizer_hash:
            raise ValueError("tokenizer SHA-256 mismatch")
        counted_tokens = 0
        for shard in manifest.get("shards", []):
            shard_path = Path(shard["path"])
            if not shard_path.is_absolute():
                shard_path = path.parent / shard_path
            tokens = int(shard["tokens"])
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            expected_bytes = tokens * 2
            actual_bytes = shard_path.stat().st_size
            if actual_bytes != expected_bytes:
                raise ValueError(
                    f"{shard_path.name} is {actual_bytes} bytes; expected {expected_bytes}"
                )
            counted_tokens += tokens
        if not manifest.get("shards"):
            raise ValueError("manifest has no shards")
        if counted_tokens != int(manifest.get("total_tokens", -1)):
            raise ValueError("total_tokens does not equal the shard sum")
        return True, f"{counted_tokens:,} tokens in {len(manifest['shards'])} shards"
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, str(error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate hardware, files, and training configuration")
    parser.add_argument("--config", type=Path, default=Path("configs/train_125m.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python", sys.version_info[:2] in {(3, 11), (3, 12)}, sys.version.split()[0]))
    checks.append(("CUDA available", torch.cuda.is_available(), str(torch.cuda.is_available())))
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        architecture = f"sm_{capability[0]}{capability[1]}"
        checks.extend(
            [
                ("GPU", True, torch.cuda.get_device_name(0)),
                ("Compute capability", capability >= (8, 0), architecture),
                ("PyTorch GPU kernel", architecture in torch.cuda.get_arch_list(), architecture),
                ("BF16", torch.cuda.is_bf16_supported(), str(torch.cuda.is_bf16_supported())),
            ]
        )
    tokenizer_path = config.paths.tokenizer / "tokenizer.json"
    checks.append(("Tokenizer", tokenizer_path.exists(), str(tokenizer_path)))
    if tokenizer_path.exists():
        tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        for name, path in (
            ("Train corpus", config.paths.train_manifest),
            ("Validation corpus", config.paths.validation_manifest),
        ):
            if path.exists():
                passed, detail = validate_manifest(path, tokenizer_hash)
                checks.append((name, passed, detail))
            else:
                checks.append((name, False, f"missing {path}"))
    disk_path = config.project.output_dir.parent
    while not disk_path.exists() and disk_path != disk_path.parent:
        disk_path = disk_path.parent
    free = shutil.disk_usage(disk_path).free
    checks.append(("Free disk", free >= 20 * 2**30, f"{free / 2**30:.1f} GiB"))
    print(json.dumps({"torch": torch.__version__, "cuda_runtime": torch.version.cuda}, indent=2))
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL':4}  {name:20} {detail}")
    if not all(passed for _, passed, _ in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

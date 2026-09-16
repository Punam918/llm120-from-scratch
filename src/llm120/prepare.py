from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from array import array
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


_TOKENIZER = None
_BOS_ID = None
_EOS_ID = None
FORMAT_VERSION = 2


@dataclass(frozen=True)
class PrepareTask:
    source: str
    output_dir: str
    tokenizer_path: str
    tokenizer_hash: str
    validation_fraction: float
    min_chars: int
    max_chunk_chars: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize Parquet documents into resumable uint16 shards")
    parser.add_argument("--input", type=Path, default=Path("data/raw/smollm2-135m-10b"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer-32k"))
    parser.add_argument("--output", type=Path, default=Path("data/tokenized"))
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--validation-fraction", type=float, default=0.001)
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--max-chunk-chars", type=int, default=1_000_000)
    return parser.parse_args()


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER, _BOS_ID, _EOS_ID
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    # Documents are intentionally streamed into fixed blocks later; avoid misleading per-document
    # warnings when an individual document exceeds the model context window.
    _TOKENIZER.model_max_length = 1_000_000_000
    _BOS_ID = _TOKENIZER.bos_token_id
    _EOS_ID = _TOKENIZER.eos_token_id
    if _BOS_ID is None or _EOS_ID is None:
        raise ValueError("Tokenizer must have BOS and EOS tokens")
    if len(_TOKENIZER) > 65_535:
        raise ValueError("uint16 output requires a tokenizer vocabulary below 65,536")


def _is_validation(text: str, fraction: float) -> bool:
    prefix = text[:16_384].encode("utf-8", errors="replace")
    digest = hashlib.blake2b(prefix + str(len(text)).encode(), digest_size=8).digest()
    value = int.from_bytes(digest, "little") / 2**64
    return value < fraction


def _flush(handle, buffer: array) -> int:
    count = len(buffer)
    if count:
        if os.sys.byteorder != "little":
            buffer.byteswap()
        buffer.tofile(handle)
        del buffer[:]
    return count


def _process(task: PrepareTask) -> dict:
    source = Path(task.source)
    output_dir = Path(task.output_dir)
    source_key = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:10]
    stem = f"{source.stem}-{source_key}"
    train_path = output_dir / f"{stem}.train.bin"
    validation_path = output_dir / f"{stem}.validation.bin"
    meta_path = output_dir / f"{stem}.meta.json"
    source_stat = source.stat()
    fingerprint = {
        "format_version": FORMAT_VERSION,
        "source": str(source.resolve()),
        "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "tokenizer_sha256": task.tokenizer_hash,
        "validation_fraction": task.validation_fraction,
        "min_chars": task.min_chars,
    }
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(old.get(key) == value for key, value in fingerprint.items()):
            if (
                train_path.exists()
                and validation_path.exists()
                and train_path.stat().st_size == old["train_tokens"] * 2
                and validation_path.stat().st_size == old["validation_tokens"] * 2
            ):
                return old | {"status": "cached"}

    train_tmp = train_path.with_suffix(train_path.suffix + ".tmp")
    validation_tmp = validation_path.with_suffix(validation_path.suffix + ".tmp")
    train_tokens = validation_tokens = documents = skipped = 0
    train_buffer = array("H")
    validation_buffer = array("H")
    parquet = pq.ParquetFile(source)
    if "text" not in parquet.schema.names:
        raise ValueError(f"{source} has no text column")
    with train_tmp.open("wb") as train_file, validation_tmp.open("wb") as validation_file:
        for batch in parquet.iter_batches(batch_size=512, columns=["text"], use_threads=False):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str) or len(text.strip()) < task.min_chars:
                    skipped += 1
                    continue
                target = validation_buffer if _is_validation(text, task.validation_fraction) else train_buffer
                target.append(_BOS_ID)
                for start in range(0, len(text), task.max_chunk_chars):
                    ids = _TOKENIZER.encode(
                        text[start : start + task.max_chunk_chars], add_special_tokens=False
                    )
                    target.extend(ids)
                target.append(_EOS_ID)
                documents += 1
                if len(train_buffer) >= 1_000_000:
                    train_tokens += _flush(train_file, train_buffer)
                if len(validation_buffer) >= 250_000:
                    validation_tokens += _flush(validation_file, validation_buffer)
        train_tokens += _flush(train_file, train_buffer)
        validation_tokens += _flush(validation_file, validation_buffer)
        train_file.flush()
        validation_file.flush()
        os.fsync(train_file.fileno())
        os.fsync(validation_file.fileno())
    os.replace(train_tmp, train_path)
    os.replace(validation_tmp, validation_path)
    metadata = fingerprint | {
        "train_path": train_path.name,
        "validation_path": validation_path.name,
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
        "documents": documents,
        "skipped_documents": skipped,
        "status": "built",
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _tokenizer_hash(path: Path) -> str:
    return hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()


def _write_manifest(output: Path, split: str, results: list[dict], tokenizer_hash: str) -> None:
    key = f"{split}_tokens"
    path_key = f"{split}_path"
    shards = [
        {"path": item[path_key], "tokens": item[key], "source": item["source"]}
        for item in sorted(results, key=lambda value: value["source"])
        if item[key] > 0
    ]
    manifest = {
        "format_version": FORMAT_VERSION,
        "dtype": "uint16",
        "byte_order": "little",
        "split": split,
        "tokenizer_sha256": tokenizer_hash,
        "total_tokens": sum(item["tokens"] for item in shards),
        "shards": shards,
    }
    (output / f"{split}-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 0.1:
        raise ValueError("--validation-fraction must be between 0 and 0.1")
    files = sorted(args.input.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found below {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer_hash = _tokenizer_hash(args.tokenizer)
    tasks = [
        PrepareTask(
            source=str(path),
            output_dir=str(args.output.resolve()),
            tokenizer_path=str(args.tokenizer.resolve()),
            tokenizer_hash=tokenizer_hash,
            validation_fraction=args.validation_fraction,
            min_chars=args.min_chars,
            max_chunk_chars=args.max_chunk_chars,
        )
        for path in files
    ]
    context = mp.get_context("spawn")
    results = []
    with context.Pool(
        processes=args.workers, initializer=_init_worker, initargs=(str(args.tokenizer.resolve()),)
    ) as pool:
        for index, result in enumerate(pool.imap_unordered(_process, tasks), start=1):
            results.append(result)
            print(
                f"[{index}/{len(tasks)}] {Path(result['source']).name}: "
                f"{result['train_tokens']:,} train + {result['validation_tokens']:,} validation "
                f"tokens ({result['status']})",
                flush=True,
            )
    _write_manifest(args.output, "train", results, tokenizer_hash)
    _write_manifest(args.output, "validation", results, tokenizer_hash)
    total = sum(item["train_tokens"] + item["validation_tokens"] for item in results)
    print(f"Prepared {total:,} tokens in {args.output.resolve()}")


if __name__ == "__main__":
    main()

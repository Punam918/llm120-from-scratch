from __future__ import annotations

import argparse
import json
import math
import random
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from llm120.prepare import _is_validation


ENGLISH_SOURCES = {"dclm_edu", "fineweb_edu", "cosmopedia_v2"}
CODE_SOURCES = {"stack_edu"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two tokenizers on paired held-out text")
    parser.add_argument("--tokenizer-a", type=Path, required=True)
    parser.add_argument("--tokenizer-b", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-domain", type=int, default=50_000)
    parser.add_argument("--tail-start-row", type=int, default=90_000)
    parser.add_argument("--max-chars", type=int, default=16_384)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=120)
    return parser.parse_args()


def _domain(source: str) -> str | None:
    if source in ENGLISH_SOURCES:
        return "english"
    if source in CODE_SOURCES:
        return "code"
    return None


def sample_tail_documents(
    files: list[Path], target: int, tail_start: int, max_chars: int, seed: int
) -> dict[str, list[str]]:
    """Sample equal per-shard quotas from rows well beyond tokenizer-training prefixes."""
    samples = {"english": [], "code": []}
    base, remainder = divmod(target, len(files))
    for file_index, path in enumerate(files):
        quota = base + (1 if file_index < remainder else 0)
        local = {"english": [], "code": []}
        parquet = pq.ParquetFile(path)
        starts: list[int] = []
        cursor = 0
        for group in range(parquet.num_row_groups):
            starts.append(cursor)
            cursor += parquet.metadata.row_group(group).num_rows
        groups = [
            group
            for group, start in enumerate(starts)
            if start + parquet.metadata.row_group(group).num_rows > tail_start
        ]
        rng = random.Random(seed + file_index * 1_000_003)
        rng.shuffle(groups)
        for group in groups:
            table = parquet.read_row_group(group, columns=["text", "source"])
            rows = list(range(table.num_rows))
            rng.shuffle(rows)
            texts = table.column("text")
            sources = table.column("source")
            for row in rows:
                if starts[group] + row < tail_start:
                    continue
                text = texts[row].as_py()
                source = sources[row].as_py()
                domain = _domain(source)
                if domain is None or len(local[domain]) >= quota:
                    continue
                if not isinstance(text, str) or len(text.strip()) < 64:
                    continue
                if _is_validation(text, 0.001):
                    continue
                normalized = unicodedata.normalize("NFC", text[:max_chars])
                local[domain].append(normalized)
                if all(len(local[name]) >= quota for name in local):
                    break
            if all(len(local[name]) >= quota for name in local):
                break
        for domain in samples:
            if len(local[domain]) != quota:
                raise RuntimeError(
                    f"{path.name}: found {len(local[domain])}/{quota} held-out {domain} documents"
                )
            samples[domain].extend(local[domain])
    for domain, texts in samples.items():
        if len(texts) != target:
            raise RuntimeError(f"Collected {len(texts)}/{target} {domain} documents")
    return samples


@dataclass
class DomainMetrics:
    token_counts: list[int] = field(default_factory=list)
    total_tokens: int = 0
    unknown_tokens: int = 0
    roundtrip_failures: int = 0
    encode_seconds: float = 0.0
    used_ids: set[int] = field(default_factory=set)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def evaluate(
    tokenizer_paths: list[Path], samples: dict[str, list[str]], batch_size: int
) -> tuple[list[Any], list[dict[str, DomainMetrics]]]:
    tokenizers = [AutoTokenizer.from_pretrained(path, use_fast=True) for path in tokenizer_paths]
    for tokenizer in tokenizers:
        tokenizer.model_max_length = 1_000_000_000
    results = [{domain: DomainMetrics() for domain in samples} for _ in tokenizers]

    # Warm both implementations before timing, then alternate order per batch to limit cache bias.
    warmup = samples["english"][:batch_size]
    for tokenizer in tokenizers:
        tokenizer(warmup, add_special_tokens=False, padding=False, truncation=False)

    batch_index = 0
    for domain, texts in samples.items():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            order = (0, 1) if batch_index % 2 == 0 else (1, 0)
            for tokenizer_index in order:
                tokenizer = tokenizers[tokenizer_index]
                metrics = results[tokenizer_index][domain]
                started = time.perf_counter()
                ids_batch = tokenizer(
                    batch, add_special_tokens=False, padding=False, truncation=False
                )["input_ids"]
                metrics.encode_seconds += time.perf_counter() - started
                decoded = tokenizer.batch_decode(
                    ids_batch,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for text, ids, reconstructed in zip(batch, ids_batch, decoded, strict=True):
                    count = len(ids)
                    metrics.token_counts.append(count)
                    metrics.total_tokens += count
                    metrics.unknown_tokens += ids.count(tokenizer.unk_token_id)
                    metrics.used_ids.update(ids)
                    if reconstructed != text:
                        metrics.roundtrip_failures += 1
            batch_index += 1
    return tokenizers, results


def summarize_domain(
    tokenizer: Any, metrics: DomainMetrics, texts: list[str]
) -> dict[str, int | float]:
    characters = sum(len(text) for text in texts)
    utf8_bytes = sum(len(text.encode("utf-8")) for text in texts)
    words = sum(len(text.split()) for text in texts)
    fertility = [
        count * 100 / max(1, len(text))
        for count, text in zip(metrics.token_counts, texts, strict=True)
    ]
    return {
        "documents": len(texts),
        "characters": characters,
        "utf8_bytes": utf8_bytes,
        "words": words,
        "tokens": metrics.total_tokens,
        "tokens_per_100_characters": metrics.total_tokens * 100 / characters,
        "characters_per_token": characters / metrics.total_tokens,
        "bytes_per_token": utf8_bytes / metrics.total_tokens,
        "tokens_per_word": metrics.total_tokens / words,
        "estimated_characters_per_2048_context": characters / metrics.total_tokens * 2048,
        "documents_over_2048_tokens_percent": sum(
            count > 2048 for count in metrics.token_counts
        )
        * 100
        / len(texts),
        "document_fertility_p50": percentile(fertility, 0.50),
        "document_fertility_p90": percentile(fertility, 0.90),
        "document_fertility_p99": percentile(fertility, 0.99),
        "unknown_tokens": metrics.unknown_tokens,
        "roundtrip_failures": metrics.roundtrip_failures,
        "observed_vocabulary_ids": len(metrics.used_ids),
        "observed_vocabulary_percent": len(metrics.used_ids) * 100 / len(tokenizer),
        "encode_seconds": metrics.encode_seconds,
        "encode_mib_per_second": utf8_bytes / 2**20 / metrics.encode_seconds,
    }


def comparison(
    a: DomainMetrics, b: DomainMetrics, summary_a: dict[str, Any], summary_b: dict[str, Any]
) -> dict[str, int | float]:
    a_wins = sum(x < y for x, y in zip(a.token_counts, b.token_counts, strict=True))
    b_wins = sum(y < x for x, y in zip(a.token_counts, b.token_counts, strict=True))
    ties = len(a.token_counts) - a_wins - b_wins
    return {
        "a_fewer_tokens_documents": a_wins,
        "b_fewer_tokens_documents": b_wins,
        "equal_token_documents": ties,
        "b_token_reduction_vs_a_percent": (
            summary_a["tokens"] - summary_b["tokens"]
        )
        * 100
        / summary_a["tokens"],
        "b_context_capacity_gain_vs_a_percent": (
            summary_b["characters_per_token"] / summary_a["characters_per_token"] - 1
        )
        * 100,
        "b_encoding_speed_change_vs_a_percent": (
            summary_b["encode_mib_per_second"] / summary_a["encode_mib_per_second"] - 1
        )
        * 100,
    }


def markdown_report(report: dict[str, Any]) -> str:
    names = report["tokenizers"]
    lines = [
        "# Tokenizer held-out evaluation",
        "",
        f"- A: `{names['a']['path']}` ({names['a']['training_documents']:,} training documents)",
        f"- B: `{names['b']['path']}` ({names['b']['training_documents']:,} training documents)",
        f"- Evaluation: {report['sampling']['samples_per_domain']:,} English + "
        f"{report['sampling']['samples_per_domain']:,} Stack-Edu code documents",
        f"- Each document capped at {report['sampling']['max_chars']:,} characters; "
        f"sampled after row {report['sampling']['tail_start_row']:,} in every shard",
        "",
    ]
    for domain in ("english", "code"):
        a = report["results"]["a"][domain]
        b = report["results"]["b"][domain]
        c = report["comparison"][domain]
        lines.extend(
            [
                f"## {domain.title()}",
                "",
                "| Metric | A | B |",
                "|---|---:|---:|",
                f"| Total tokens | {a['tokens']:,} | {b['tokens']:,} |",
                f"| Tokens / 100 chars | {a['tokens_per_100_characters']:.4f} | "
                f"{b['tokens_per_100_characters']:.4f} |",
                f"| Characters / token | {a['characters_per_token']:.4f} | "
                f"{b['characters_per_token']:.4f} |",
                f"| Tokens / word | {a['tokens_per_word']:.4f} | {b['tokens_per_word']:.4f} |",
                f"| Estimated chars / 2048 context | "
                f"{a['estimated_characters_per_2048_context']:.0f} | "
                f"{b['estimated_characters_per_2048_context']:.0f} |",
                f"| Documents >2048 tokens | {a['documents_over_2048_tokens_percent']:.3f}% | "
                f"{b['documents_over_2048_tokens_percent']:.3f}% |",
                f"| Fertility p50 / p90 / p99 | {a['document_fertility_p50']:.2f} / "
                f"{a['document_fertility_p90']:.2f} / {a['document_fertility_p99']:.2f} | "
                f"{b['document_fertility_p50']:.2f} / {b['document_fertility_p90']:.2f} / "
                f"{b['document_fertility_p99']:.2f} |",
                f"| Encoding MiB/s | {a['encode_mib_per_second']:.2f} | "
                f"{b['encode_mib_per_second']:.2f} |",
                f"| Observed vocabulary | {a['observed_vocabulary_percent']:.2f}% | "
                f"{b['observed_vocabulary_percent']:.2f}% |",
                f"| Unknown tokens | {a['unknown_tokens']} | {b['unknown_tokens']} |",
                f"| Round-trip failures | {a['roundtrip_failures']} | "
                f"{b['roundtrip_failures']} |",
                "",
                f"B reduces tokens by **{c['b_token_reduction_vs_a_percent']:.4f}%** versus A. "
                f"Paired document wins: A {c['a_fewer_tokens_documents']:,}, "
                f"B {c['b_fewer_tokens_documents']:,}, ties {c['equal_token_documents']:,}.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    files = sorted(args.data.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet shards below {args.data}")
    samples = sample_tail_documents(
        files,
        target=args.samples_per_domain,
        tail_start=args.tail_start_row,
        max_chars=args.max_chars,
        seed=args.seed,
    )
    paths = [args.tokenizer_a.resolve(), args.tokenizer_b.resolve()]
    tokenizers, metrics = evaluate(paths, samples, args.batch_size)
    summaries: list[dict[str, dict[str, Any]]] = []
    for tokenizer, tokenizer_metrics in zip(tokenizers, metrics, strict=True):
        summaries.append(
            {
                domain: summarize_domain(tokenizer, tokenizer_metrics[domain], samples[domain])
                for domain in samples
            }
        )
    metadata = [
        json.loads((path / "training-metadata.json").read_text(encoding="utf-8"))
        for path in paths
    ]
    report = {
        "tokenizers": {
            "a": {"path": str(paths[0]), **metadata[0]},
            "b": {"path": str(paths[1]), **metadata[1]},
        },
        "sampling": {
            "dataset": str(args.data.resolve()),
            "shards": len(files),
            "samples_per_domain": args.samples_per_domain,
            "tail_start_row": args.tail_start_row,
            "max_chars": args.max_chars,
            "seed": args.seed,
            "english_sources": sorted(ENGLISH_SOURCES),
            "code_sources": sorted(CODE_SOURCES),
        },
        "results": {"a": summaries[0], "b": summaries[1]},
        "comparison": {
            domain: comparison(
                metrics[0][domain],
                metrics[1][domain],
                summaries[0][domain],
                summaries[1][domain],
            )
            for domain in samples
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    rendered = markdown_report(report)
    (args.output / "report.md").write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

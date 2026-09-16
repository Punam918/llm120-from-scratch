from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = [
    "<|beginoftext|>",
    "<|endoftext|>",
    "<|padding|>",
    "<|unknown|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]

CHAT_TEMPLATE = """{%- for message in messages -%}
{%- if loop.first -%}{{ bos_token }}{%- endif -%}
{%- if message['role'] == 'system' -%}{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}
{%- elif message['role'] == 'user' -%}{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}
{%- elif message['role'] == 'assistant' -%}{{ '<|assistant|>\n' }}{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}{{ '\n' }}
{%- else -%}{{ raise_exception('Unsupported chat role: ' + message['role']) }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{ '<|assistant|>\n' }}{%- endif -%}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer from scratch")
    parser.add_argument("--input", type=Path, default=Path("data/raw/smollm2-135m-10b"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/tokenizer-32k"))
    parser.add_argument("--vocab-size", type=int, default=32_768)
    parser.add_argument("--max-documents", type=int, default=500_000)
    parser.add_argument("--max-chars-per-document", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parquet_files(root: Path) -> list[Path]:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found below {root}")
    return files


def text_iterator(
    files: list[Path], max_documents: int, max_chars: int, seed: int
) -> Iterator[str]:
    paths = files.copy()
    random.Random(seed).shuffle(paths)
    emitted = 0
    per_file_limit = math.ceil(max_documents / len(paths))
    while emitted < max_documents:
        made_progress = False
        for path in paths:
            emitted_from_file = 0
            parquet = pq.ParquetFile(path)
            if "text" not in parquet.schema.names:
                continue
            for batch in parquet.iter_batches(batch_size=1024, columns=["text"]):
                for text in batch.column(0).to_pylist():
                    if isinstance(text, str) and len(text.strip()) >= 64:
                        yield text[:max_chars]
                        emitted += 1
                        emitted_from_file += 1
                        made_progress = True
                        if emitted >= max_documents:
                            return
                        if emitted_from_file >= per_file_limit:
                            break
                if emitted_from_file >= per_file_limit:
                    break
        if not made_progress:
            raise RuntimeError("No usable text rows were found")
        # Only repeat for deliberately tiny pilot datasets.
        random.Random(seed + emitted).shuffle(paths)


def build_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[3]))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def main() -> None:
    args = parse_args()
    files = parquet_files(args.input)
    tokenizer = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    print(f"Training {args.vocab_size:,}-token byte BPE on up to {args.max_documents:,} documents")
    tokenizer.train_from_iterator(
        text_iterator(files, args.max_documents, args.max_chars_per_document, args.seed),
        trainer=trainer,
        length=args.max_documents,
    )
    bos_id = tokenizer.token_to_id(SPECIAL_TOKENS[0])
    eos_id = tokenizer.token_to_id(SPECIAL_TOKENS[1])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[(SPECIAL_TOKENS[0], bos_id), (SPECIAL_TOKENS[1], eos_id)],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=SPECIAL_TOKENS[0],
        eos_token=SPECIAL_TOKENS[1],
        pad_token=SPECIAL_TOKENS[2],
        unk_token=SPECIAL_TOKENS[3],
        model_max_length=2048,
        clean_up_tokenization_spaces=False,
        additional_special_tokens=SPECIAL_TOKENS[4:],
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(args.output)
    probes = [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n: int) -> int:",
        "नेपाल एक सुन्दर देश हो।",
    ]
    report = {}
    for probe in probes:
        ids = fast.encode(probe, add_special_tokens=False)
        decoded = fast.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        report[probe] = {"tokens": len(ids), "unknown_tokens": ids.count(fast.unk_token_id)}
        if decoded != normalizers.NFC().normalize_str(probe):
            raise RuntimeError(f"Tokenizer round-trip failed for {probe!r}: {decoded!r}")
    metadata = {
        "vocab_size": len(fast),
        "training_documents": args.max_documents,
        "input_files": len(files),
        "seed": args.seed,
        "probes": report,
    }
    (args.output / "training-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

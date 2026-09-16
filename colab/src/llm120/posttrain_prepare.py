from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


SYSTEM_PROMPT = "You are a helpful, honest, and concise AI assistant."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and normalize assistant-tuning data")
    parser.add_argument("--input", type=Path, default=Path("data/posttrain/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/posttrain/normalized"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-rows", type=int, default=2_000)
    return parser.parse_args()


def parquet_files(root: Path) -> list[str]:
    files = sorted(str(path) for path in root.rglob("*.parquet") if "/.cache/" not in str(path))
    if not files:
        raise FileNotFoundError(f"No Parquet data found under {root}")
    return files


def load_sft(root: Path, seed: int, validation_rows: int) -> DatasetDict:
    files = parquet_files(root)
    train = [path for path in files if "train" in Path(path).name.lower()]
    validation = [
        path
        for path in files
        if any(key in Path(path).name.lower() for key in ("test", "validation", "eval"))
    ]
    if not train:
        train = files
    data_files: dict[str, list[str]] = {"train": train}
    if validation:
        data_files["validation"] = validation
    loaded = load_dataset("parquet", data_files=data_files)
    if "validation" not in loaded:
        size = min(validation_rows, max(1, len(loaded["train"]) // 100))
        loaded = loaded["train"].train_test_split(test_size=size, seed=seed)
        loaded = DatasetDict(train=loaded["train"], validation=loaded["test"])
    return DatasetDict(train=loaded["train"], validation=loaded["validation"])


def valid_messages(messages: Any) -> bool:
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    allowed = {"system", "user", "assistant"}
    return all(
        isinstance(item, dict)
        and item.get("role") in allowed
        and isinstance(item.get("content"), str)
        and bool(item["content"].strip())
        for item in messages
    ) and messages[-1]["role"] == "assistant"


def explicit_preference(row: dict[str, Any]) -> dict[str, Any]:
    chosen, rejected = row["chosen"], row["rejected"]
    common = 0
    for chosen_message, rejected_message in zip(chosen, rejected, strict=False):
        if chosen_message != rejected_message:
            break
        common += 1
    if common == 0:
        raise ValueError("Preference pair has no shared prompt")
    return {
        "prompt": chosen[:common],
        "chosen": chosen[common:],
        "rejected": rejected[common:],
    }


def load_dpo(root: Path, seed: int, validation_rows: int) -> DatasetDict:
    loaded = load_dataset("parquet", data_files={"train": parquet_files(root)})["train"]
    needed = {"chosen", "rejected"}
    if not needed.issubset(loaded.column_names):
        raise ValueError(f"DPO data is missing columns {sorted(needed - set(loaded.column_names))}")
    loaded = loaded.filter(lambda row: valid_messages(row["chosen"]) and valid_messages(row["rejected"]))

    loaded = loaded.map(explicit_preference, remove_columns=loaded.column_names)
    loaded = loaded.filter(
        lambda row: bool(row["prompt"])
        and bool(row["chosen"])
        and bool(row["rejected"])
        and row["chosen"][-1]["role"] == "assistant"
        and row["rejected"][-1]["role"] == "assistant"
    )
    size = min(validation_rows, max(1, len(loaded) // 100))
    split = loaded.train_test_split(test_size=size, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"])


def helpsteer_file(root: Path) -> Path:
    candidates = sorted(
        path
        for path in root.rglob("preference*.jsonl*")
        if path.is_file()
        and path.stat().st_size > 0
        and not any(part.startswith(".cache") for part in path.parts)
    )
    if not candidates:
        candidates = sorted(
            path
            for path in root.rglob("*.jsonl*")
            if path.is_file()
            and path.stat().st_size > 0
            and "preference" in str(path)
            and not any(part.startswith(".cache") for part in path.parts)
        )
    if not candidates:
        raise FileNotFoundError(f"HelpSteer2 preference JSONL was not found under {root}")
    return candidates[0]


def normalize_helpsteer(row: dict[str, Any]) -> dict[str, Any]:
    strength = int(row["preference_strength"])
    response1 = row.get("response_1", row.get("response1"))
    response2 = row.get("response_2", row.get("response2"))
    chosen_text, rejected_text = (response1, response2) if strength < 0 else (response2, response1)
    prompt = row["prompt"]
    prefix = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return {
        "chosen": [{"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
        "prompt": prefix,
        "preference_strength": abs(strength),
        "source_split": str(row.get("split", "train")),
    }


def load_helpsteer(root: Path, seed: int, validation_rows: int) -> tuple[DatasetDict, DatasetDict]:
    cache_dir = root.parent / ".datasets-cache" / "helpsteer2-preference"
    data = load_dataset(
        "json", data_files={"train": str(helpsteer_file(root))}, cache_dir=str(cache_dir)
    )["train"]
    required = {"prompt", "preference_strength"}
    if not required.issubset(data.column_names):
        raise ValueError(f"HelpSteer2 data is missing {sorted(required - set(data.column_names))}")
    data = data.filter(
        lambda row: int(row["preference_strength"]) != 0
        and bool(str(row["prompt"]).strip())
        and bool(str(row.get("response_1", row.get("response1", ""))).strip())
        and bool(str(row.get("response_2", row.get("response2", ""))).strip())
    )
    normalized = data.map(normalize_helpsteer, remove_columns=data.column_names)
    validation = normalized.filter(lambda row: row["source_split"] in {"validation", "test"})
    train = normalized.filter(lambda row: row["source_split"] == "train")
    if not len(train) or not len(validation):
        size = min(validation_rows, max(1, len(normalized) // 10))
        split = normalized.train_test_split(test_size=size, seed=seed)
        train, validation = split["train"], split["test"]
    reward = DatasetDict(
        train=train.remove_columns("source_split"),
        validation=validation.remove_columns("source_split"),
    )

    def unique_prompts(dataset: Dataset) -> Dataset:
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for row in dataset:
            key = json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                rows.append({"prompt": row["prompt"]})
        return Dataset.from_list(rows)

    rlhf = DatasetDict(
        train=unique_prompts(reward["train"]), validation=unique_prompts(reward["validation"])
    )
    return reward, rlhf


def write_parquet(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    dataset.to_parquet(temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    sft = load_sft(args.input / "smol-smoltalk", args.seed, args.validation_rows)
    for split in sft:
        if "messages" not in sft[split].column_names:
            raise ValueError("Smol-Smoltalk must contain a messages column")
        sft[split] = sft[split].filter(lambda row: valid_messages(row["messages"]))
    dpo = load_dpo(args.input / "ultrafeedback-cleaned", args.seed, args.validation_rows)
    reward, rlhf = load_helpsteer(args.input / "helpsteer2", args.seed, args.validation_rows)

    collections = {"sft": sft, "dpo": dpo, "reward": reward, "rlhf": rlhf}
    counts: dict[str, dict[str, int]] = {}
    for name, dataset_dict in collections.items():
        counts[name] = {}
        for split, dataset in dataset_dict.items():
            write_parquet(dataset, args.output / f"{name}-{split}.parquet")
            counts[name][split] = len(dataset)

    manifest = {
        "seed": args.seed,
        "validation_rows_requested": args.validation_rows,
        "system_prompt": SYSTEM_PROMPT,
        "rows": counts,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

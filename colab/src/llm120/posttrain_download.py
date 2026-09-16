from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    repo_id: str
    directory: str
    purpose: str


DATASETS = (
    DatasetSpec(
        "sft",
        "HuggingFaceTB/smol-smoltalk",
        "smol-smoltalk",
        "supervised instruction tuning",
    ),
    DatasetSpec(
        "dpo",
        "argilla/ultrafeedback-binarized-preferences-cleaned",
        "ultrafeedback-cleaned",
        "offline preference optimization",
    ),
    DatasetSpec(
        "reward",
        "nvidia/HelpSteer2",
        "helpsteer2",
        "human preference reward modeling and online RLHF prompts",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the pinned assistant-tuning datasets")
    parser.add_argument("--output", type=Path, default=Path("data/posttrain/raw"))
    parser.add_argument(
        "--only", choices=["all", *(item.key for item in DATASETS)], default="all"
    )
    return parser.parse_args()


def repo_size(info: object) -> int:
    return sum(int(getattr(item, "size", 0) or 0) for item in getattr(info, "siblings", []))


def main() -> None:
    args = parse_args()
    selected = DATASETS if args.only == "all" else tuple(x for x in DATASETS if x.key == args.only)
    args.output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    manifest: dict[str, object] = {"datasets": {}}

    for spec in selected:
        info = api.dataset_info(spec.repo_id, files_metadata=True)
        required = repo_size(info)
        free = shutil.disk_usage(args.output).free
        if required and free < required + 5 * 1024**3:
            raise RuntimeError(
                f"Not enough free space for {spec.repo_id}: need {required / 1024**3:.1f} GiB "
                "plus a 5 GiB safety margin"
            )
        target = args.output / spec.directory
        print(f"Downloading {spec.repo_id} ({required / 1024**2:.1f} MiB) -> {target}")
        snapshot_download(
            repo_id=spec.repo_id,
            repo_type="dataset",
            revision=info.sha,
            local_dir=target,
        )
        manifest["datasets"][spec.key] = {
            "repo_id": spec.repo_id,
            "revision": info.sha,
            "bytes": required,
            "path": str(target),
            "purpose": spec.purpose,
        }

    manifest_path = args.output / "download-manifest.json"
    previous: dict[str, object] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous.setdefault("datasets", {})
    previous["datasets"].update(manifest["datasets"])
    manifest_path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

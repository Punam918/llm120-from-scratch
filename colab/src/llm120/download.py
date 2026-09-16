from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


DEFAULT_DATASET = "EleutherAI/SmolLM2-135M-10B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a pinned set of Parquet data shards")
    parser.add_argument("--repo-id", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("data/raw/smollm2-135m-10b"))
    parser.add_argument("--pattern", default="data/*.parquet")
    parser.add_argument(
        "--max-shards",
        type=int,
        help="Download an evenly spaced subset for a pilot run; omit for the full dataset",
    )
    return parser.parse_args()


def evenly_spaced(items: list[str], count: int) -> list[str]:
    if count >= len(items):
        return items
    if count < 1:
        raise ValueError("--max-shards must be positive")
    if count == 1:
        return [items[len(items) // 2]]
    indices = {round(i * (len(items) - 1) / (count - 1)) for i in range(count)}
    return [items[index] for index in sorted(indices)]


def main() -> None:
    args = parse_args()
    api = HfApi()
    info = api.dataset_info(args.repo_id, revision=args.revision, files_metadata=True)
    files = sorted(
        sibling.rfilename
        for sibling in info.siblings
        if sibling.rfilename.startswith(args.pattern.split("*")[0])
        and sibling.rfilename.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"No Parquet files matched {args.pattern!r} in {args.repo_id}")
    selected = evenly_spaced(files, args.max_shards) if args.max_shards else files
    sizes = {sibling.rfilename: (sibling.size or 0) for sibling in info.siblings}
    download_bytes = sum(sizes[name] for name in selected)
    args.output.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output.resolve()).free
    # Leave ample space for uint16 tokens, checkpoints, and filesystem headroom.
    if download_bytes and free_bytes < download_bytes * 3:
        raise RuntimeError(
            f"Only {free_bytes / 2**30:.1f} GiB free for a {download_bytes / 2**30:.1f} GiB "
            "download. Keep at least 3x the compressed size free."
        )
    print(
        f"Downloading {len(selected)}/{len(files)} shards "
        f"({download_bytes / 2**30:.1f} GiB) at revision {info.sha}"
    )
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=info.sha,
        allow_patterns=selected + ["README.md"],
        local_dir=args.output,
    )
    manifest = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": info.sha,
        "files": selected,
        "compressed_bytes": download_bytes,
    }
    (args.output / "download-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Download complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()

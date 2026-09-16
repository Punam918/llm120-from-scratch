from __future__ import annotations

import bisect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class TokenShard:
    path: Path
    tokens: int
    blocks: int


class TokenBlockCorpus:
    """Random-access fixed-length blocks backed by uint16 memory maps.

    Training order uses an affine permutation. It is stateless, covers every block
    exactly once per epoch, and therefore resumes at an exact data position without
    storing a multi-gigabyte shuffled index.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        sequence_length: int,
        *,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dtype") != "uint16":
            raise ValueError(f"Expected a uint16 manifest, got {manifest.get('dtype')!r}")
        self.sequence_length = sequence_length
        self.seed = seed
        self.shuffle = shuffle
        self.shards: list[TokenShard] = []
        self._arrays: dict[int, np.memmap] = {}
        cumulative = []
        total = 0
        for item in manifest["shards"]:
            path = Path(item["path"])
            if not path.is_absolute():
                path = (self.manifest_path.parent / path).resolve()
            tokens = int(item["tokens"])
            blocks = tokens // sequence_length
            if blocks:
                self.shards.append(TokenShard(path=path, tokens=tokens, blocks=blocks))
                total += blocks
                cumulative.append(total)
        if total == 0:
            raise ValueError(f"No complete {sequence_length}-token blocks in {manifest_path}")
        self._cumulative_blocks = cumulative
        self.num_blocks = total

    def _permutation(self, epoch: int) -> tuple[int, int]:
        rng = random.Random(self.seed + epoch * 1_000_003)
        multiplier = rng.randrange(1, self.num_blocks)
        while math.gcd(multiplier, self.num_blocks) != 1:
            multiplier = (multiplier + 1) % self.num_blocks or 1
        return multiplier, rng.randrange(self.num_blocks)

    def permuted_index(self, sample_index: int) -> int:
        epoch, index = divmod(sample_index, self.num_blocks)
        if not self.shuffle:
            return index
        multiplier, offset = self._permutation(epoch)
        return (multiplier * index + offset) % self.num_blocks

    def _array(self, shard_index: int) -> np.memmap:
        if shard_index not in self._arrays:
            shard = self.shards[shard_index]
            expected_bytes = shard.tokens * np.dtype("<u2").itemsize
            actual_bytes = shard.path.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"Corrupt shard {shard.path}: expected {expected_bytes} bytes, got {actual_bytes}"
                )
            self._arrays[shard_index] = np.memmap(shard.path, dtype="<u2", mode="r")
        return self._arrays[shard_index]

    def block(self, sample_index: int) -> torch.Tensor:
        block_index = self.permuted_index(sample_index)
        shard_index = bisect.bisect_right(self._cumulative_blocks, block_index)
        previous = 0 if shard_index == 0 else self._cumulative_blocks[shard_index - 1]
        local_block = block_index - previous
        start = local_block * self.sequence_length
        # Copy detaches the tensor from the read-only memmap and avoids PyTorch warnings.
        values = np.array(
            self._array(shard_index)[start : start + self.sequence_length],
            dtype=np.int64,
            copy=True,
        )
        return torch.from_numpy(values)

    def batch(self, first_sample: int, batch_size: int) -> torch.Tensor:
        return torch.stack([self.block(first_sample + index) for index in range(batch_size)])

    @property
    def usable_tokens(self) -> int:
        return self.num_blocks * self.sequence_length


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


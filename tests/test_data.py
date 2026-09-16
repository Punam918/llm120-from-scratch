import json
from pathlib import Path

import numpy as np

from llm120.data import TokenBlockCorpus


def make_manifest(tmp_path: Path, values: np.ndarray) -> Path:
    shard = tmp_path / "tokens.bin"
    values.astype("<u2").tofile(shard)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "shards": [{"path": shard.name, "tokens": len(values)}],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_blocks_are_a_full_permutation_each_epoch(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, np.arange(48, dtype=np.uint16))
    corpus = TokenBlockCorpus(manifest, sequence_length=4, seed=7, shuffle=True)
    first = [corpus.permuted_index(index) for index in range(corpus.num_blocks)]
    second = [
        corpus.permuted_index(index + corpus.num_blocks) for index in range(corpus.num_blocks)
    ]
    assert sorted(first) == list(range(corpus.num_blocks))
    assert sorted(second) == list(range(corpus.num_blocks))
    assert first != second


def test_block_contents_and_resume_are_stateless(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, np.arange(32, dtype=np.uint16))
    one = TokenBlockCorpus(manifest, sequence_length=4, seed=11, shuffle=True)
    two = TokenBlockCorpus(manifest, sequence_length=4, seed=11, shuffle=True)
    assert one.block(13).tolist() == two.block(13).tolist()


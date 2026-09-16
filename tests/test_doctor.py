import hashlib
import json

from llm120.doctor import validate_manifest


def test_validate_manifest_checks_tokenizer_and_binary_size(tmp_path) -> None:
    tokenizer_hash = hashlib.sha256(b"tokenizer").hexdigest()
    shard = tmp_path / "train.bin"
    shard.write_bytes(b"\x00\x00\x01\x00")
    manifest = tmp_path / "train-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dtype": "uint16",
                "byte_order": "little",
                "tokenizer_sha256": tokenizer_hash,
                "total_tokens": 2,
                "shards": [{"path": shard.name, "tokens": 2}],
            }
        )
    )
    assert validate_manifest(manifest, tokenizer_hash)[0]
    assert not validate_manifest(manifest, "wrong-hash")[0]
    shard.write_bytes(b"\x00\x00")
    assert not validate_manifest(manifest, tokenizer_hash)[0]

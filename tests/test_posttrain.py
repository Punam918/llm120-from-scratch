from llm120.posttrain_common import linear_warmup_steps
from llm120.posttrain_prepare import explicit_preference, helpsteer_file, normalize_helpsteer
from llm120.rlhf import completion_health_reward


def test_explicit_preference_removes_shared_prompt() -> None:
    user = {"role": "user", "content": "Question"}
    row = {
        "chosen": [user, {"role": "assistant", "content": "Good"}],
        "rejected": [user, {"role": "assistant", "content": "Bad"}],
    }
    normalized = explicit_preference(row)
    assert normalized["prompt"] == [user]
    assert normalized["chosen"][0]["content"] == "Good"
    assert normalized["rejected"][0]["content"] == "Bad"


def test_helpsteer_preference_sign() -> None:
    base = {
        "prompt": "Question",
        "response_1": "First",
        "response_2": "Second",
        "split": "validation",
    }
    negative = normalize_helpsteer(base | {"preference_strength": -2})
    positive = normalize_helpsteer(base | {"preference_strength": 2})
    assert negative["chosen"][0]["content"] == "First"
    assert positive["chosen"][0]["content"] == "Second"
    assert positive["source_split"] == "validation"


def test_completion_health_penalizes_repetition() -> None:
    clean = [[{"role": "assistant", "content": "A concise useful answer."}]]
    repeated = [[{"role": "assistant", "content": "again and again and again and again and again"}]]
    assert completion_health_reward(clean, [[1]])[0] > completion_health_reward(repeated, [[1]])[0]


def test_helpsteer_file_ignores_hub_lock(tmp_path) -> None:
    lock = tmp_path / ".cache" / "huggingface" / "preference.jsonl.gz.lock"
    lock.parent.mkdir(parents=True)
    lock.touch()
    data = tmp_path / "preference" / "preference.jsonl.gz"
    data.parent.mkdir()
    data.write_bytes(b"data")
    assert helpsteer_file(tmp_path) == data


def test_linear_warmup_uses_override_or_dataset_size() -> None:
    assert linear_warmup_steps(
        rows=1_000, epochs=2.0, batch_size=2, accumulation_steps=5, max_steps=-1
    ) == 20
    assert linear_warmup_steps(
        rows=1_000, epochs=2.0, batch_size=2, accumulation_steps=5, max_steps=250
    ) == 25
    assert linear_warmup_steps(
        rows=3, epochs=1.0, batch_size=2, accumulation_steps=8, max_steps=1
    ) == 0

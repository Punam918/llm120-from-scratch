import pytest

from llm120.schedule import wsd_factor


def test_wsd_schedule() -> None:
    values = [
        wsd_factor(step, max_steps=100, warmup_steps=10, decay_ratio=0.2)
        for step in range(100)
    ]
    assert values[0] == pytest.approx(0.1)
    assert values[9] == pytest.approx(1.0)
    assert values[79] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.0)
    assert all(left >= right for left, right in zip(values[79:], values[80:]))


from __future__ import annotations


def wsd_factor(
    step: int,
    *,
    max_steps: int,
    warmup_steps: int,
    decay_ratio: float,
    min_ratio: float = 0.0,
) -> float:
    """Warmup-stable-decay factor for the next optimizer update (zero-indexed)."""
    if step < warmup_steps:
        return max(min_ratio, (step + 1) / max(1, warmup_steps))
    decay_steps = max(1, round(max_steps * decay_ratio))
    decay_start = max(warmup_steps, max_steps - decay_steps)
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start + 1) / max(1, max_steps - decay_start))
    return min_ratio + (1.0 - min_ratio) * (1.0 - progress)


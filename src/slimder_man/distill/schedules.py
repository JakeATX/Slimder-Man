from __future__ import annotations

import math


def linear_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    t = min(max(step, 0), total_steps - 1) / (total_steps - 1)
    return start + (end - start) * t


def cosine_schedule(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return end
    t = min(max(step, 0), total_steps - 1) / (total_steps - 1)
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * t))


def global_cosine_lr(base_lr: float, min_lr: float, warmup_steps: int, step: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    denom = max(1, total_steps - warmup_steps)
    t = min(max(step - warmup_steps, 0), denom) / denom
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))

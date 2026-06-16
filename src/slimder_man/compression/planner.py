from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StagePlan:
    stage: int
    remove_last_n_layers: int
    hidden_size: int
    routed_experts: int | None
    top_k: int | None
    tokens: int


def progressive_plan(schedule: str, stages: int, total_tokens: int, token_split: list[float], teacher_layers: int, remove_last_n: int, start_hidden: int, target_hidden: int, hidden_multiple: int = 128, target_experts: int | None = None, target_top_k: int | None = None) -> list[StagePlan]:
    if stages == 1 or schedule == "one_stage":
        return [StagePlan(1, remove_last_n, target_hidden, target_experts, target_top_k, int(total_tokens * token_split[0]))]
    half_depth = remove_last_n // 2
    half_hidden = start_hidden - (((start_hidden - target_hidden) // 2) // hidden_multiple) * hidden_multiple
    if schedule == "depth_first":
        stage1 = StagePlan(1, half_depth, start_hidden, None, None, int(total_tokens * token_split[0]))
    elif schedule == "width_first":
        stage1 = StagePlan(1, 0, half_hidden, None, None, int(total_tokens * token_split[0]))
    elif schedule == "joint":
        stage1 = StagePlan(1, half_depth, half_hidden, None, None, int(total_tokens * token_split[0]))
    else:
        raise ValueError(f"Unsupported progressive schedule {schedule}")
    stage2 = StagePlan(2, remove_last_n, target_hidden, target_experts, target_top_k, total_tokens - stage1.tokens)
    return [stage1, stage2]

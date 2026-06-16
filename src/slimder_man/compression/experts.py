from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ExpertMergePlan:
    s_keep: list[int]
    s_base: list[int]
    groups: dict[int, list[int]]
    new_expert_order: list[int]
    warning: str | None = None


def partial_preservation_plan(scores: torch.Tensor, similarity: torch.Tensor, target_experts: int) -> ExpertMergePlan:
    n = int(scores.numel())
    if target_experts <= 0 or target_experts >= n:
        raise ValueError("target expert count must be between 1 and original expert count - 1")
    order = torch.argsort(scores, descending=True, stable=True).tolist()
    keep_n = target_experts // 2
    s_keep = order[:keep_n]
    remaining = [i for i in order if i not in s_keep]
    base_n = target_experts - keep_n
    s_base = remaining[:base_n]
    assign = [i for i in range(n) if i not in set(s_keep + s_base)]
    groups = {b: [] for b in s_base}
    for i in assign:
        best = max(s_base, key=lambda b: float(similarity[i, b]))
        groups[best].append(i)
    return ExpertMergePlan(s_keep=s_keep, s_base=s_base, groups=groups, new_expert_order=s_keep + s_base)


def _weighted_average_state(experts: list[nn.Module], weights: torch.Tensor) -> nn.Module:
    out = deepcopy(experts[0])
    finite = torch.isfinite(weights) & (weights > 0)
    warning = not bool(finite.any())
    if warning:
        weights = torch.ones_like(weights, dtype=torch.float64)
    else:
        weights = torch.where(finite, torch.clamp(weights.to(torch.float64), min=1e-12), torch.zeros_like(weights, dtype=torch.float64))
    weights = weights / weights.sum()
    states = [e.state_dict() for e in experts]
    merged = {}
    for key in states[0]:
        tensor = sum(w.to(states[i][key].device, states[i][key].dtype) * states[i][key] for i, w in enumerate(weights))
        merged[key] = tensor
    out.load_state_dict(merged)
    return out


def merge_experts(experts: list[nn.Module], scores: torch.Tensor, similarity: torch.Tensor, target_experts: int) -> tuple[list[nn.Module], ExpertMergePlan]:
    plan = partial_preservation_plan(scores, similarity, target_experts)
    new_experts = [deepcopy(experts[i]) for i in plan.s_keep]
    warning = None
    for base in plan.s_base:
        indices = [base] + plan.groups[base]
        group_scores = scores[indices]
        if not bool((torch.isfinite(group_scores) & (group_scores > 0)).any()):
            warning = "all merge scores were zero or nonfinite; used uniform weights"
        new_experts.append(_weighted_average_state([experts[i] for i in indices], group_scores))
    plan.warning = warning
    return new_experts, plan

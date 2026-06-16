import torch
from torch import nn

from slimder_man.compression.experts import merge_experts, partial_preservation_plan


def test_partial_preservation_merge_plan_and_weights():
    experts = [nn.Linear(1, 1, bias=False) for _ in range(8)]
    for i, expert in enumerate(experts):
        expert.weight.data.fill_(float(i))
    scores = torch.tensor([10, 9, 8, 7, 1, 1, 1, 1], dtype=torch.float32)
    sim = torch.eye(8)
    sim[4, 2] = sim[5, 2] = 1
    sim[6, 3] = sim[7, 3] = 1
    plan = partial_preservation_plan(scores, sim, 4)
    assert plan.s_keep == [0, 1]
    assert plan.s_base == [2, 3]
    merged, plan = merge_experts(experts, scores, sim, 4)
    assert len(merged) == 4
    assert merged[0].weight.item() == experts[0].weight.item()
    expected_base2 = (8 * 2 + 1 * 4 + 1 * 5) / 10
    assert abs(merged[2].weight.item() - expected_base2) < 1e-6
    covered = set(plan.s_keep + plan.s_base + sum(plan.groups.values(), []))
    assert covered == set(range(8))


def test_zero_scores_fall_back_uniform():
    experts = [nn.Linear(1, 1, bias=False) for _ in range(4)]
    scores = torch.zeros(4)
    sim = torch.eye(4)
    _, plan = merge_experts(experts, scores, sim, 2)
    assert plan.warning

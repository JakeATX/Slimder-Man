import torch

from slimder_man.calibration.stats import expert_frequency, expert_reap_importance, expert_soft_importance


def test_expert_metrics_match_crafted_assignments():
    topi = torch.tensor([[0, 1], [1, 2], [1, 3]])
    topw = torch.tensor([[0.75, 0.25], [0.6, 0.4], [0.9, 0.1]])
    norm2 = torch.zeros(3, 4)
    norm2[:, 0] = 2
    norm2[:, 1] = 3
    norm2[:, 2] = 5
    norm2[:, 3] = 7
    freq = expert_frequency(topi, 4)
    soft = expert_soft_importance(topi, topw, 4)
    reap = expert_reap_importance(topi, topw, norm2, 4)
    assert torch.allclose(freq, torch.tensor([1 / 3, 1.0, 1 / 3, 1 / 3], dtype=torch.float64))
    assert torch.allclose(soft, torch.tensor([0.25, (0.25 + 0.6 + 0.9) / 3, 0.4 / 3, 0.1 / 3], dtype=torch.float64))
    assert reap[0] == soft[0] * 2
    assert not torch.isnan(reap).any()


def test_unselected_expert_zero_not_nan():
    topi = torch.tensor([[0], [0]])
    topw = torch.ones(2, 1)
    norm2 = torch.ones(2, 3)
    assert expert_soft_importance(topi, topw, 3)[2] == 0

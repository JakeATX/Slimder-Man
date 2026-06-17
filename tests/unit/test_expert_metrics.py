import torch

from slimder_man.calibration.stats import expert_frequency, expert_reap_importance, expert_soft_importance
from slimder_man.calibration.stats import expert_reap_numerator_counts, finalize_reap_importance


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
    assert torch.allclose(
        reap,
        torch.tensor([
            0.75 * 2,
            ((0.25 * 3) + (0.6 * 3) + (0.9 * 3)) / 3,
            0.4 * 5,
            0.1 * 7,
        ], dtype=torch.float64),
    )
    assert not torch.isnan(reap).any()


def test_unselected_expert_zero_not_nan():
    topi = torch.tensor([[0], [0]])
    topw = torch.ones(2, 1)
    norm2 = torch.ones(2, 3)
    assert expert_soft_importance(topi, topw, 3)[2] == 0


def test_reap_uses_assigned_token_mean_not_total_token_mean():
    topi = torch.tensor([[0], [0], [1], [2]])
    topw = torch.ones(4, 1)
    norm2 = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [0.0, 12.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )

    reap = expert_reap_importance(topi, topw, norm2, 3)

    assert torch.allclose(reap, torch.tensor([4.0, 12.0, 20.0], dtype=torch.float64))


def test_reap_global_aggregation_does_not_dilute_absent_batches():
    batch1_topi = torch.tensor([[0], [0]])
    batch1_topw = torch.ones(2, 1)
    batch1_norm = torch.tensor([[2.0, 0.0], [6.0, 0.0]])
    batch2_topi = torch.tensor([[1], [1]])
    batch2_topw = torch.ones(2, 1)
    batch2_norm = torch.tensor([[0.0, 10.0], [0.0, 14.0]])

    num1, count1 = expert_reap_numerator_counts(batch1_topi, batch1_topw, batch1_norm, 2)
    num2, count2 = expert_reap_numerator_counts(batch2_topi, batch2_topw, batch2_norm, 2)
    global_reap = finalize_reap_importance(num1 + num2, count1 + count2)

    assert torch.allclose(global_reap, torch.tensor([4.0, 12.0], dtype=torch.float64))

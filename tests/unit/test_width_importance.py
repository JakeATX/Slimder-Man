import torch
import pytest

from slimder_man.compression.width import select_keep_indices


def test_width_keep_indices_are_sorted_and_deterministic():
    scores = torch.zeros(16)
    scores[[3, 7, 9, 11]] = torch.tensor([10.0, 9.0, 8.0, 7.0])
    keep1 = select_keep_indices(scores, 4)
    keep2 = select_keep_indices(scores, 4)
    assert keep1.tolist() == [3, 7, 9, 11]
    assert torch.equal(keep1, keep2)


def test_bad_target_hidden_size_raises():
    with pytest.raises(ValueError):
        select_keep_indices(torch.ones(4), 0)

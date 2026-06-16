import pytest
import torch

from slimder_man.compression.router import router_rows_for_merge


def test_router_rows_base_strategy():
    rows = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    out = router_rows_for_merge(rows, [0, 1], [3, 4])
    assert torch.equal(out, rows[[0, 1, 3, 4]])


def test_weighted_average_rejected():
    with pytest.raises(ValueError):
        router_rows_for_merge(torch.zeros(2, 2), [0], [1], strategy="weighted_average")

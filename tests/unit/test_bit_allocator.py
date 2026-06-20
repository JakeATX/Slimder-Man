import pytest

from slimder_man.quant.bit_allocator import QuantItem, allocate_bits


def test_allocator_budget_protection_and_monotonicity():
    items = [QuantItem("hot", 10, 10), QuantItem("cold", 10, 1), QuantItem("router", 5, 99, protected_bits=16)]
    out = allocate_bits(items, [2, 4, 8, 16], target_avg_bits=8)
    assert out["router"] == 16
    assert out["hot"] >= out["cold"]
    assert sum(i.size * out[i.name] for i in items) <= sum(i.size for i in items) * 8


def test_infeasible_budget_fails():
    with pytest.raises(ValueError) as exc:
        allocate_bits([QuantItem("router", 10, 1, protected_bits=16)], [2, 4], 4)
    message = str(exc.value)
    assert "minimum_feasible_avg_bits" in message
    assert "router(size=10, protected_bits=16)" in message

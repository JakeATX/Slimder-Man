import pytest

from slimder_man.compression.planner import estimate_parameters, validate_target


def architecture() -> dict:
    return {
        "total_params": 100_000,
        "active_params_estimate": 60_000,
        "hidden_size": 16,
        "vocab_size": 257,
        "num_layers": 4,
        "moe_layers": [{"layer_idx": idx, "num_routed_experts": 8, "num_shared_experts": 1, "top_k": 2} for idx in range(4)],
        "tied_embeddings": False,
    }


def test_parameter_estimate_scales_total_active_and_memory():
    estimate = estimate_parameters(architecture(), hidden_size=12, remove_last_n_layers=1, routed_experts=4, routed_top_k=2)

    assert estimate.total_params > 0
    assert estimate.active_params > 0
    assert estimate.active_params <= estimate.total_params
    assert estimate.memory_bytes == estimate.total_params * 2
    assert estimate.memory_gib == round(estimate.memory_bytes / 1024**3, 3)


def test_parameter_estimate_reflects_more_active_experts():
    smaller_top_k = estimate_parameters(architecture(), 12, 1, 4, 1)
    larger_top_k = estimate_parameters(architecture(), 12, 1, 4, 2)

    assert larger_top_k.total_params == smaller_top_k.total_params
    assert larger_top_k.active_params > smaller_top_k.active_params


def test_validate_target_rejects_invalid_constraints():
    arch = architecture()

    with pytest.raises(ValueError, match="top_k must not exceed routed_experts"):
        validate_target(arch, hidden_size=12, remove_last_n_layers=1, routed_experts=1, routed_top_k=2, hidden_multiple=4)
    with pytest.raises(ValueError, match="hidden multiple"):
        validate_target(arch, hidden_size=10, remove_last_n_layers=1, routed_experts=4, routed_top_k=2, hidden_multiple=4)
    with pytest.raises(ValueError, match="leave at least one layer"):
        validate_target(arch, hidden_size=12, remove_last_n_layers=4, routed_experts=4, routed_top_k=2, hidden_multiple=4)

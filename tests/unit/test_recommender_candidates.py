import pytest

from slimder_man.analyze.plans import apply_recommendation_to_config, architecture_fingerprint, validate_applied_plan
from slimder_man.analyze.recommender import recommend
from slimder_man.config.schema import SlimderConfig


def qwen_architecture() -> dict:
    return {
        "model_type": "qwen3_next",
        "total_params": 32_000_000_000,
        "active_params_estimate": 3_000_000_000,
        "hidden_size": 2048,
        "vocab_size": 151_936,
        "num_layers": 48,
        "moe_layers": [{"layer_idx": idx, "num_routed_experts": 512, "num_shared_experts": 1, "top_k": 10} for idx in range(48)],
        "has_mtp": False,
        "mtp_depths": 0,
        "tied_embeddings": False,
    }


def test_qwen_anchor_is_first_and_candidates_are_diverse():
    candidates = recommend(qwen_architecture(), "slimqwen_anchor")

    assert len(candidates) >= 3
    anchor = candidates[0]
    assert anchor["hidden_size"] == 1536
    assert anchor["remove_last_n_layers"] == 12
    assert anchor["routed_experts"] == 256
    assert anchor["routed_top_k"] == 8
    assert anchor["schedule"] == "depth_first"
    assert anchor["transforms"]["width"] == {
        "from_hidden_size": 2048,
        "to_hidden_size": 1536,
        "hidden_multiple": 128,
    }
    assert anchor["transforms"]["depth"] == {"from_layers": 48, "remove_last_n_layers": 12, "to_layers": 36}
    assert anchor["estimated_total_params"] > 0
    assert anchor["estimated_active_params"] > 0
    assert anchor["estimated_memory_bytes"] == anchor["estimated_total_params"] * 2
    assert anchor["estimated_memory_gib"] > 0

    assert len({candidate["hidden_size"] for candidate in candidates}) > 1
    assert len({candidate["remove_last_n_layers"] for candidate in candidates}) > 1
    assert len({candidate["routed_experts"] for candidate in candidates}) > 1
    assert len({candidate["routed_top_k"] for candidate in candidates}) > 1


def test_candidates_obey_target_constraints():
    for candidate in recommend(qwen_architecture(), "slimqwen_anchor"):
        assert candidate["hidden_size"] % 128 == 0
        assert 0 <= candidate["remove_last_n_layers"] < 48
        assert 0 < candidate["routed_experts"] <= 512
        assert 0 < candidate["routed_top_k"] <= 10
        assert candidate["routed_top_k"] <= candidate["routed_experts"]


def test_all_preset_groups_return_multiple_valid_candidates():
    candidates = recommend(qwen_architecture(), "all", max_candidates=3)
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["preset"], []).append(candidate)

    assert set(groups) == {"conservative_20", "balanced_50", "slimqwen_anchor", "aggressive_80", "extreme_90"}
    assert all(len(group) >= 3 for group in groups.values())
    assert all(candidate["candidate_id"].startswith(candidate["preset"]) for candidate in candidates)


def test_qwen_non_anchor_presets_solve_for_param_reduction_not_tiny_constants():
    expected_reductions = {
        "conservative_20": 0.20,
        "balanced_50": 0.50,
        "aggressive_80": 0.80,
        "extreme_90": 0.90,
    }

    for preset, target in expected_reductions.items():
        first = recommend(qwen_architecture(), preset, max_candidates=3)[0]
        assert first["target_total_param_reduction"] == target
        assert first["total_reduction_error"] <= 0.01
        assert first["estimated_total_param_reduction"] == pytest.approx(target, abs=0.01)
        assert first["reduction_solver"] == "estimated_param_grid"
        assert first["hidden_size"] >= 1024
        assert first["routed_experts"] > 0
        assert first["routed_top_k"] <= first["routed_experts"]


def test_config_param_reduction_override_drives_recommendation_plan():
    cfg = SlimderConfig(
        teacher={"load_mode": "transformers", "model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"},
        compression={"preset": "balanced_50", "target": {"total_param_reduction": 0.65}},
    )
    cfg, plan = apply_recommendation_to_config(cfg, qwen_architecture(), preset="balanced_50")

    candidate = plan["candidate"]
    assert candidate["target_total_param_reduction"] == 0.65
    assert candidate["estimated_total_param_reduction"] == pytest.approx(0.65, abs=0.01)
    assert cfg.compression.target.total_param_reduction == 0.65
    assert cfg.compression.plan is not None
    assert cfg.compression.plan.target.total_param_reduction == 0.65


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        recommend(qwen_architecture(), "missing")


def test_apply_recommendation_materializes_qwen_anchor_target_and_fingerprint():
    cfg = SlimderConfig(teacher={"load_mode": "transformers", "model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"})
    cfg, plan = apply_recommendation_to_config(cfg, qwen_architecture(), preset="slimqwen_anchor", candidate_id="slimqwen_anchor_1")

    assert cfg.compression.target.hidden_size == 1536
    assert cfg.compression.target.remove_last_n_layers == 12
    assert cfg.compression.target.routed_experts == 256
    assert cfg.compression.target.routed_top_k == 8
    assert cfg.compression.plan is not None
    assert cfg.compression.plan.source_architecture_fingerprint == architecture_fingerprint(qwen_architecture())
    assert plan["candidate"]["candidate_id"] == "slimqwen_anchor_1"
    validate_applied_plan(cfg, qwen_architecture())


def test_applied_plan_rejects_architecture_and_target_mismatches():
    cfg = SlimderConfig(project={"paper_faithful": False})
    cfg, _ = apply_recommendation_to_config(cfg, qwen_architecture(), preset="slimqwen_anchor", candidate_id="slimqwen_anchor_1")
    changed_arch = {**qwen_architecture(), "hidden_size": 4096}

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_applied_plan(cfg, changed_arch)

    cfg.compression.target.hidden_size = 1792
    with pytest.raises(ValueError, match="does not match compression.plan target"):
        validate_applied_plan(cfg, qwen_architecture())

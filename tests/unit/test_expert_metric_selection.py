import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.calibration.collectors import CalibrationResult
from slimder_man.compression.apply import compress_tiny_model
from slimder_man.config.schema import SlimderConfig


def _calibration(freq: torch.Tensor, soft: torch.Tensor, sim: torch.Tensor) -> CalibrationResult:
    return CalibrationResult(
        hidden_scores=torch.arange(16, dtype=torch.float32),
        per_layer_hidden_scores=[torch.ones(16) for _ in range(9)],
        expert_frequency=[freq.clone() for _ in range(4)],
        expert_soft=[soft.clone() for _ in range(4)],
        expert_reap=[soft.clone() for _ in range(4)],
        expert_similarity=[sim.clone() for _ in range(4)],
    )


def _cfg(**expert_updates) -> SlimderConfig:
    return SlimderConfig(
        project={"paper_faithful": False},
        compression={
            "target": {"hidden_size": 16, "remove_last_n_layers": 0, "routed_experts": 4, "routed_top_k": 2},
            "experts": expert_updates,
        },
    )


def test_configured_importance_metric_changes_kept_experts():
    teacher = TinyMoEForCausalLM()
    freq = torch.tensor([8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.float32)
    soft = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.float32)
    cal = _calibration(freq, soft, torch.eye(8))

    _, freq_manifest = compress_tiny_model(teacher, _cfg(importance_metric="frequency"), cal)
    _, soft_manifest = compress_tiny_model(teacher, _cfg(importance_metric="soft_logits"), cal)

    assert freq_manifest["experts"]["layers"][0]["s_keep"] == [0, 1]
    assert soft_manifest["experts"]["layers"][0]["s_keep"] == [7, 6]


def test_similarity_matrix_changes_merge_assignment():
    teacher = TinyMoEForCausalLM()
    scores = torch.tensor([8, 7, 6, 5, 1, 1, 1, 1], dtype=torch.float32)
    sim_to_2 = torch.eye(8)
    sim_to_2[4, 2] = 10
    sim_to_3 = torch.eye(8)
    sim_to_3[4, 3] = 10

    _, manifest_2 = compress_tiny_model(teacher, _cfg(importance_metric="frequency"), _calibration(scores, scores, sim_to_2))
    _, manifest_3 = compress_tiny_model(teacher, _cfg(importance_metric="frequency"), _calibration(scores, scores, sim_to_3))

    assert 4 in manifest_2["experts"]["layers"][0]["groups"]["2"]
    assert 4 in manifest_3["experts"]["layers"][0]["groups"]["3"]


def test_prune_method_keeps_top_scored_experts_without_merge_groups():
    teacher = TinyMoEForCausalLM()
    scores = torch.tensor([1, 9, 3, 8, 2, 7, 4, 6], dtype=torch.float32)
    _, manifest = compress_tiny_model(teacher, _cfg(method="prune", importance_metric="frequency"), _calibration(scores, scores, torch.eye(8)))

    first_layer = manifest["experts"]["layers"][0]
    assert first_layer["s_keep"] == [1, 3, 5, 7]
    assert first_layer["s_base"] == []
    assert first_layer["groups"] == {}

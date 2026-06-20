import pytest
from pydantic import ValidationError

from slimder_man.config.schema import SlimderConfig


def test_valid_paper_faithful_config_parses():
    assert SlimderConfig().project.paper_faithful is True


def test_paper_faithful_rejects_unsupported_depth_method():
    with pytest.raises(ValidationError, match="depth.method=last_layers"):
        SlimderConfig(compression={"depth": {"method": "activation_similarity"}})


def test_paper_faithful_rejects_topk_logit_cache():
    with pytest.raises(ValidationError, match="offline_topk_logit_cache"):
        SlimderConfig(kd={"teacher_mode": "offline_topk_logit_cache"})


def test_paper_faithful_rejects_shared_expert_quant_prune():
    with pytest.raises(ValidationError, match="shared expert"):
        SlimderConfig(quantization={"prune_shared_experts": True})
    with pytest.raises(ValidationError, match="keeping shared experts"):
        SlimderConfig(compression={"experts": {"keep_shared_experts": False}})


def test_paper_faithful_rejects_non_default_schedules_and_expert_prune():
    with pytest.raises(ValidationError, match="lambda"):
        SlimderConfig(kd={"lambda_schedule": {"type": "constant", "start": 1.0, "end": 1.0}})
    with pytest.raises(ValidationError, match="beta"):
        SlimderConfig(kd={"mtp": {"beta_schedule": {"type": "linear", "start": 0.3, "end": 0.1}}})
    with pytest.raises(ValidationError, match="partial_preservation_merge"):
        SlimderConfig(compression={"experts": {"method": "prune"}})


def test_augmented_config_accepts_saliency_quantization():
    cfg = SlimderConfig(project={"paper_faithful": False}, quantization={"enabled": True})
    assert cfg.quantization.enabled


def test_expert_outputs_similarity_is_rejected_until_real_output_sketches_exist():
    with pytest.raises(ValidationError, match="similarity_metric"):
        SlimderConfig(project={"paper_faithful": False}, compression={"experts": {"similarity_metric": "expert_outputs"}})


def test_augmented_config_accepts_topk_logit_cache_path():
    cfg = SlimderConfig(
        project={"paper_faithful": False},
        kd={"teacher_mode": "offline_topk_logit_cache", "offline_topk_logits_cache_path": "topk_cache.pt"},
    )
    assert cfg.kd.teacher_mode == "offline_topk_logit_cache"
    assert cfg.kd.offline_topk_logits_cache_path == "topk_cache.pt"


def test_nested_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        SlimderConfig(compression={"depth": {"methdo": "last_layers"}})


def test_param_reduction_targets_are_accepted_for_planning_and_bounded():
    cfg = SlimderConfig(compression={"target": {"total_param_reduction": 0.5, "active_param_reduction": 0.4}})
    assert cfg.compression.target.total_param_reduction == 0.5
    assert cfg.compression.target.active_param_reduction == 0.4
    with pytest.raises(ValidationError, match="total_param_reduction"):
        SlimderConfig(compression={"target": {"total_param_reduction": 1.0}})
    with pytest.raises(ValidationError, match="active_param_reduction"):
        SlimderConfig(compression={"target": {"active_param_reduction": 0.0}})


def test_depth_remove_fraction_schema_bounds():
    assert SlimderConfig(compression={"target": {"depth_remove_fraction": 0.25}}).compression.target.depth_remove_fraction == 0.25
    with pytest.raises(ValidationError, match="depth_remove_fraction"):
        SlimderConfig(compression={"target": {"depth_remove_fraction": 1.0}})

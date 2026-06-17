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


def test_nested_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        SlimderConfig(compression={"depth": {"methdo": "last_layers"}})


def test_unimplemented_param_reduction_targets_are_rejected():
    with pytest.raises(ValidationError, match="total_param_reduction"):
        SlimderConfig(compression={"target": {"total_param_reduction": 0.5}})
    with pytest.raises(ValidationError, match="active_param_reduction"):
        SlimderConfig(compression={"target": {"active_param_reduction": 0.5}})


def test_depth_remove_fraction_schema_bounds():
    assert SlimderConfig(compression={"target": {"depth_remove_fraction": 0.25}}).compression.target.depth_remove_fraction == 0.25
    with pytest.raises(ValidationError, match="depth_remove_fraction"):
        SlimderConfig(compression={"target": {"depth_remove_fraction": 1.0}})

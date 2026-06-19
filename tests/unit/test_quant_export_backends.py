import pytest

from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM
from slimder_man.quant import awq_adapter, bnb_adapter, gptq_adapter, smoothquant_adapter
from slimder_man.quant.backend_adapters import OptionalQuantBackendUnavailable
from slimder_man.quant.export import collect_export_hashes
from slimder_man.quant.fake_backend import fake_quantize_model
from slimder_man.quant.sensitivity import protected_bits_for_name, quant_items_from_sensitivity, sensitivity_records_for_model
from slimder_man.utils.json import read_json


def test_sensitivity_records_include_protection_and_quant_error():
    model = DummyHfMoeForCausalLM()
    records = sensitivity_records_for_model(model)
    by_name = {record.name: record for record in records}

    router = by_name["model.layers.0.mlp.gate.weight"]
    expert = by_name["model.layers.0.mlp.experts.0.up_proj.weight"]
    assert protected_bits_for_name(router.name) == 16
    assert router.protected_bits == 16
    assert router.signals["quant_error"] is not None
    assert expert.protected_bits is None
    assert expert.saliency >= 0
    items = quant_items_from_sensitivity(records)
    assert next(item for item in items if item.name == router.name).protected_bits == 16


def test_fake_quant_writes_export_manifest_with_hashes(tmp_path):
    model = DummyHfMoeForCausalLM()
    manifest = fake_quantize_model(model, tmp_path, target_avg_bits=12.0, safe_serialization=True)
    export_manifest = read_json(tmp_path / "quant_export_manifest.json")

    assert manifest["export_manifest"] == "quant_export_manifest.json"
    assert manifest == read_json(tmp_path / "fake_quant_manifest.json")
    assert export_manifest["backend"] == "fake_symmetric_uniform"
    assert export_manifest["backend_manifest"]["allocation"] == manifest["allocation"]
    assert export_manifest["artifact_hashes"] == collect_export_hashes(tmp_path)
    assert "model.safetensors" in export_manifest["artifact_hashes"]
    assert "fake_quant_manifest.json" in export_manifest["artifact_hashes"]


def test_optional_quant_backend_status_and_actionable_unavailable_errors():
    for module, expected_backend in [
        (awq_adapter, "awq"),
        (gptq_adapter, "gptq"),
        (bnb_adapter, "bitsandbytes"),
    ]:
        status = module.backend_status()
        assert status["backend"] == expected_backend
        assert "install_hint" in status
        if not status["available"]:
            with pytest.raises(OptionalQuantBackendUnavailable, match=expected_backend):
                module.quantize(None)


def test_smoothquant_scaffold_is_explicitly_not_claimed_available():
    status = smoothquant_adapter.backend_status()
    assert status["backend"] == "smoothquant"
    assert status["available"] is False
    with pytest.raises(NotImplementedError, match="not implemented"):
        smoothquant_adapter.quantize(None)

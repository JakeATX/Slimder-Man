from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM
from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.quant.fake_backend import fake_quantize_model, fake_quantize_tiny_model, fake_quantize_tensor
from slimder_man.utils.json import read_json


def test_fake_quantize_tensor_limits_values_to_uniform_grid():
    import torch

    tensor = torch.tensor([-1.0, -0.3, 0.2, 1.0])
    quantized = fake_quantize_tensor(tensor, bits=4)
    assert quantized.shape == tensor.shape
    assert torch.max(torch.abs(quantized)) <= 1.0
    assert not torch.equal(quantized, tensor)


def test_fake_quant_backend_writes_loadable_tiny_artifact(tmp_path):
    model = TinyMoEForCausalLM()
    manifest = fake_quantize_tiny_model(model, tmp_path, target_avg_bits=12.0)
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "fake_quant_manifest.json").exists()
    assert manifest == read_json(tmp_path / "fake_quant_manifest.json")
    assert "embed_tokens.weight" in manifest["allocation"]
    assert manifest["validation"]["finite_loss"] is True
    assert manifest["validation"]["logits_shape"] == [1, 8, model.config.vocab_size]
    loaded = TinyMoEForCausalLM.from_pretrained(tmp_path)
    assert sum(p.numel() for p in loaded.parameters()) == sum(p.numel() for p in model.parameters())


def test_fake_quant_backend_writes_loadable_hf_dummy_artifact(tmp_path):
    model = DummyHfMoeForCausalLM()
    manifest = fake_quantize_model(model, tmp_path, target_avg_bits=12.0, safe_serialization=True)

    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "fake_quant_manifest.json").exists()
    assert manifest == read_json(tmp_path / "fake_quant_manifest.json")
    assert manifest["model_type"] == "dummy_hf_moe"
    assert manifest["allocation"]["model.layers.0.mlp.gate.weight"] == 16
    assert manifest["validation"]["finite_loss"] is True
    assert manifest["validation"]["logits_shape"] == [1, 8, model.config.vocab_size]
    loaded = DummyHfMoeForCausalLM.from_pretrained(tmp_path)
    assert sum(p.numel() for p in loaded.parameters()) == sum(p.numel() for p in model.parameters())

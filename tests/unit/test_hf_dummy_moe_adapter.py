import sys
from pathlib import Path

import torch

from slimder_man.adapters.generic_hf_moe import GenericHfMoeAdapter
from slimder_man.adapters.registry import get_adapter

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyHfMoeForCausalLM


def test_generic_hf_moe_adapter_identifies_structural_moe_layers():
    model = DummyHfMoeForCausalLM()
    adapter = get_adapter(model)

    assert isinstance(adapter, GenericHfMoeAdapter)
    layers = adapter.iter_moe_layers(model)
    assert len(layers) == model.config.num_hidden_layers
    assert len(adapter.get_routed_experts(layers[0])) == model.config.num_experts
    assert len(adapter.get_shared_experts(layers[0])) == model.config.num_shared_experts
    assert adapter.get_router(layers[0]).out_features == model.config.num_experts

    info = adapter.describe_architecture(model)
    assert info.model_type == "dummy_hf_moe"
    assert info.hidden_size == 32
    assert info.vocab_size == 257
    assert info.num_layers == 3
    assert [layer.layer_idx for layer in info.moe_layers] == [0, 1, 2]
    assert all(layer.num_routed_experts == 6 for layer in info.moe_layers)
    assert all(layer.num_shared_experts == 2 for layer in info.moe_layers)
    assert all(layer.top_k == 2 for layer in info.moe_layers)


def test_dummy_hf_moe_forward_is_non_tiny_hf_compatible():
    model = DummyHfMoeForCausalLM()
    output = model(torch.tensor([[1, 2, 3, 4]]))
    assert output.logits.shape == (1, 4, model.config.vocab_size)

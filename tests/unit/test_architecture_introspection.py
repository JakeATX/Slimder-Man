from slimder_man.adapters.registry import get_adapter
from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.adapters.base import count_parameters


def test_tiny_architecture_introspection():
    model = TinyMoEForCausalLM()
    info = get_adapter(model).describe_architecture(model)
    assert info.num_layers == 4
    assert info.moe_layers[0].num_routed_experts == 8
    assert info.moe_layers[0].num_shared_experts == 1
    assert info.moe_layers[0].top_k == 2
    assert info.hidden_size == 16
    assert info.total_params == count_parameters(model)
    assert info.has_mtp and info.mtp_depths == 2
    assert info.tied_embeddings

import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.compression.width import slice_tiny_hidden


def test_width_slicing_shapes_and_forward():
    model = TinyMoEForCausalLM()
    keep = torch.arange(12)
    slice_tiny_hidden(model, keep)
    assert model.embed_tokens.weight.shape == (128, 12)
    assert model.norm.weight.shape == (12,)
    moe = model.layers[0].moe
    assert moe.router.in_features == 12
    assert moe.experts[0].w1.in_features == 12
    assert moe.experts[0].w3.in_features == 12
    assert moe.experts[0].w2.out_features == 12
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    out = model(torch.randint(0, 128, (2, 6)))
    assert out.logits.shape == (2, 6, 128)

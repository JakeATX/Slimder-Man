import pytest
import torch

from slimder_man.adapters.tiny import TinyAdapter, TinyMoEForCausalLM
from slimder_man.compression.depth import compute_depth_keep_indices


def test_depth_pruning_keeps_last_layers_and_forward_works():
    model = TinyMoEForCausalLM()
    keep = compute_depth_keep_indices(4, 1)
    assert keep == [0, 1, 2]
    TinyAdapter().drop_blocks(model, keep)
    assert model.config.num_layers == 3
    out = model(torch.randint(0, 128, (1, 5)), labels=torch.randint(0, 128, (1, 5)))
    assert out.logits.shape[-1] == 128


def test_drop_all_layers_raises():
    with pytest.raises(ValueError):
        compute_depth_keep_indices(4, 4)

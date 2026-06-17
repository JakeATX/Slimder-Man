import pytest
import torch

from slimder_man.adapters.tiny import TinyAdapter, TinyMoEForCausalLM
from slimder_man.compression.depth import compute_depth_keep_indices, resolve_remove_last_n


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


def test_depth_remove_fraction_resolves_to_last_layer_count():
    assert resolve_remove_last_n(48, remove_last_n=0, depth_remove_fraction=0.25) == 12
    assert resolve_remove_last_n(4, remove_last_n=0, depth_remove_fraction=0.5) == 2
    assert resolve_remove_last_n(4, remove_last_n=1, depth_remove_fraction=None) == 1
    with pytest.raises(ValueError, match="depth_remove_fraction"):
        resolve_remove_last_n(4, remove_last_n=0, depth_remove_fraction=1.0)

import pytest
import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM


@pytest.mark.gpu
def test_one_gpu_tiny_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = TinyMoEForCausalLM().cuda()
    ids = torch.randint(0, 128, (1, 8), device="cuda")
    out = model(ids, labels=ids)
    assert torch.isfinite(out.loss)
    del model
    torch.cuda.empty_cache()

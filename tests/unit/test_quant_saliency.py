import pytest
from pydantic import ValidationError

from slimder_man.config.schema import SlimderConfig
from slimder_man.quant.saliency import composite_saliency


def test_saliency_renormalizes_and_shared_bonus():
    cold = composite_saliency({"freq": 0.1, "soft": 0.1, "reap": 0.1, "hessian_trace": None})
    hot = composite_saliency({"freq": 1.0, "soft": 1.0, "reap": 1.0, "hessian_trace": None})
    shared = composite_saliency({"freq": 0.1}, shared=True)
    assert hot > cold
    assert shared > hot


def test_quant_rejected_in_paper_faithful():
    with pytest.raises(ValidationError):
        SlimderConfig(quantization={"enabled": True})

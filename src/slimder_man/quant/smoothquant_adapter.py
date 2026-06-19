from __future__ import annotations

from slimder_man.quant.backend_adapters import QuantBackendSpec, unavailable_quantize


SPEC = QuantBackendSpec(
    name="smoothquant",
    module_names=("torch",),
    install_hint="SmoothQuant export currently requires a backend-specific calibration implementation.",
    mode="algorithm_scaffold",
)


def backend_status() -> dict:
    status = SPEC.status()
    status["available"] = False
    status["reason"] = "SmoothQuant algorithm scaffold is present, but packed export is not implemented in v1."
    return status


def quantize(*args, **kwargs):
    raise NotImplementedError("SmoothQuant calibration/export is scaffolded but not implemented in v1.")

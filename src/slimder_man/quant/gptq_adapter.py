from __future__ import annotations

from slimder_man.quant.backend_adapters import QuantBackendSpec, unavailable_quantize


SPEC = QuantBackendSpec(
    name="gptq",
    module_names=("auto_gptq",),
    install_hint="Install auto-gptq in an augmented quantization environment.",
    mode="optional_export_backend",
)


def backend_status() -> dict:
    return SPEC.status()


def quantize(*args, **kwargs):
    return unavailable_quantize(SPEC, *args, **kwargs)

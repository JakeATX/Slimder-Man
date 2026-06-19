from __future__ import annotations

from slimder_man.quant.backend_adapters import QuantBackendSpec, unavailable_quantize


SPEC = QuantBackendSpec(
    name="bitsandbytes",
    module_names=("bitsandbytes",),
    install_hint="Install bitsandbytes in an augmented quantization environment.",
    mode="optional_runtime_backend",
)


def backend_status() -> dict:
    return SPEC.status()


def quantize(*args, **kwargs):
    return unavailable_quantize(SPEC, *args, **kwargs)

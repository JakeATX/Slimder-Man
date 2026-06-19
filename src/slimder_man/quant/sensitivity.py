from __future__ import annotations

from dataclasses import dataclass, field

import torch

from slimder_man.quant.bit_allocator import QuantItem
from slimder_man.quant.saliency import composite_saliency


PROTECTED_MARKERS = ("router", "gate", "norm", "embed_tokens", "lm_head", "shared")


@dataclass(frozen=True)
class SensitivityRecord:
    name: str
    size: int
    saliency: float
    signals: dict[str, float | None] = field(default_factory=dict)
    protected_bits: int | None = None


def protected_bits_for_name(name: str) -> int | None:
    return 16 if any(marker in name for marker in PROTECTED_MARKERS) else None


def symmetric_uniform_quant_error(tensor: torch.Tensor, bits: int = 4) -> float:
    if not torch.is_floating_point(tensor) or tensor.numel() == 0:
        return 0.0
    data = tensor.detach().float()
    max_abs = data.abs().max()
    if max_abs == 0:
        return 0.0
    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax
    quantized = torch.clamp(torch.round(data / scale), min=-qmax, max=qmax) * scale
    return float(torch.mean((data - quantized).pow(2)).item())


def sensitivity_records_for_model(
    model: torch.nn.Module,
    extra_signals: dict[str, dict[str, float | None]] | None = None,
    error_bits: int = 4,
) -> list[SensitivityRecord]:
    records: list[SensitivityRecord] = []
    extra_signals = extra_signals or {}
    for name, param in model.named_parameters():
        signals = {
            "freq": None,
            "soft": None,
            "reap": None,
            "quant_error": symmetric_uniform_quant_error(param, bits=error_bits),
            "hessian_trace": None,
            "perplexity_delta": None,
            **extra_signals.get(name, {}),
        }
        protected = protected_bits_for_name(name)
        saliency = composite_saliency(signals, shared="shared" in name)
        records.append(
            SensitivityRecord(
                name=name,
                size=param.numel(),
                saliency=saliency,
                signals=signals,
                protected_bits=protected,
            )
        )
    return records


def quant_items_from_sensitivity(records: list[SensitivityRecord]) -> list[QuantItem]:
    return [
        QuantItem(
            name=record.name,
            size=record.size,
            saliency=record.saliency,
            protected_bits=record.protected_bits,
        )
        for record in records
    ]

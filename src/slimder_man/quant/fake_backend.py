from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.quant.bit_allocator import QuantItem, allocate_bits
from slimder_man.utils.json import write_json


def fake_quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    if not torch.is_floating_point(tensor) or tensor.numel() == 0:
        return tensor.clone()
    max_abs = tensor.detach().abs().max()
    if max_abs == 0:
        return tensor.clone()
    qmax = (2 ** (bits - 1)) - 1
    scale = max_abs / qmax
    return torch.clamp(torch.round(tensor / scale), min=-qmax, max=qmax) * scale


def _protected_bits(name: str) -> int | None:
    protected_markers = ("router", "norm", "embed_tokens", "lm_head", "shared")
    return 16 if any(marker in name for marker in protected_markers) else None


def fake_quantize_tiny_model(
    model: TinyMoEForCausalLM,
    output_dir: str | Path,
    target_avg_bits: float = 8.0,
    allowed_bits: list[int] | None = None,
) -> dict[str, Any]:
    bits = allowed_bits or [4, 8]
    items = [
        QuantItem(
            name=name,
            size=param.numel(),
            saliency=float(param.detach().abs().mean().item()),
            protected_bits=_protected_bits(name),
        )
        for name, param in model.named_parameters()
    ]
    allocation = allocate_bits(items, bits + [16], target_avg_bits)
    quantized = deepcopy(model)
    with torch.no_grad():
        for name, param in quantized.named_parameters():
            param.copy_(fake_quantize_tensor(param, allocation[name]))
    out = Path(output_dir)
    quantized.save_pretrained(out)
    manifest = {
        "backend": "fake_symmetric_uniform",
        "target_avg_bits": target_avg_bits,
        "allowed_bits": bits,
        "allocation": allocation,
        "note": "Fake quantization stores dequantized tensors for runtime smoke tests; it is not a packed production format.",
    }
    write_json(out / "fake_quant_manifest.json", manifest)
    return manifest

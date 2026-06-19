from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
from typing import Any

import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.analyze.architecture import describe_model
from slimder_man.quant.bit_allocator import QuantItem, allocate_bits
from slimder_man.utils.hashing import sha256_file
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
    protected_markers = ("router", "gate", "norm", "embed_tokens", "lm_head", "shared")
    return 16 if any(marker in name for marker in protected_markers) else None


def _save_quantized_model(model: torch.nn.Module, output_dir: Path, safe_serialization: bool) -> None:
    if isinstance(model, TinyMoEForCausalLM):
        model.save_pretrained(output_dir)
        return
    if not hasattr(model, "save_pretrained"):
        raise ValueError("Fake quantized model does not expose save_pretrained")
    save_pretrained = model.save_pretrained
    signature = inspect.signature(save_pretrained)
    accepts_safe_serialization = "safe_serialization" in signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    if accepts_safe_serialization:
        model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    else:
        model.save_pretrained(output_dir)


def fake_quantize_model(
    model: torch.nn.Module,
    output_dir: str | Path,
    target_avg_bits: float = 8.0,
    allowed_bits: list[int] | None = None,
    safe_serialization: bool = True,
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
    arch = describe_model(model)
    vocab_size = int(arch["vocab_size"])
    validation_input = torch.arange(0, min(8, vocab_size), dtype=torch.long).unsqueeze(0) % vocab_size
    with torch.no_grad():
        validation_out = (
            quantized(validation_input, labels=validation_input)
            if isinstance(quantized, TinyMoEForCausalLM)
            else quantized(input_ids=validation_input, labels=validation_input)
        )
    if validation_out.loss is None or not torch.isfinite(validation_out.loss):
        raise ValueError("Fake quantized model failed finite-loss validation")
    out = Path(output_dir)
    _save_quantized_model(quantized, out, safe_serialization=safe_serialization)
    artifact_hashes = {
        name: sha256_file(out / name)
        for name in ("model.pt", "model.safetensors", "pytorch_model.bin", "config.json")
        if (out / name).exists()
    }
    manifest = {
        "backend": "fake_symmetric_uniform",
        "model_type": arch["model_type"],
        "target_avg_bits": target_avg_bits,
        "allowed_bits": bits,
        "allocation": allocation,
        "artifact_hashes": artifact_hashes,
        "validation": {
            "finite_loss": True,
            "loss": float(validation_out.loss.detach().cpu()),
            "logits_shape": list(validation_out.logits.shape),
        },
        "note": "Fake quantization stores dequantized tensors for runtime smoke tests; it is not a packed production format.",
    }
    write_json(out / "fake_quant_manifest.json", manifest)
    return manifest


def fake_quantize_tiny_model(
    model: TinyMoEForCausalLM,
    output_dir: str | Path,
    target_avg_bits: float = 8.0,
    allowed_bits: list[int] | None = None,
) -> dict[str, Any]:
    return fake_quantize_model(model, output_dir, target_avg_bits=target_avg_bits, allowed_bits=allowed_bits)

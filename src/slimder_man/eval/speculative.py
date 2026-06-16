from __future__ import annotations

import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM


def speculative_acceptance(model: TinyMoEForCausalLM, input_ids: torch.Tensor) -> dict[str, float]:
    out = model(input_ids)
    base = out.logits.argmax(dim=-1)
    acc = {}
    for i, mtp in enumerate(out.mtp_logits):
        if mtp.shape[1] <= i + 1:
            continue
        draft = mtp[:, : -(i + 1), :].argmax(dim=-1)
        verify = base[:, i + 1 :]
        acc[f"acc_{i}"] = float((draft == verify).float().mean())
    return acc

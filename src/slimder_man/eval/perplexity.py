from __future__ import annotations

import math

import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM


def tiny_perplexity(model: TinyMoEForCausalLM, batches: list[torch.Tensor]) -> float:
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            out = model(batch, labels=batch)
            if out.loss is not None:
                losses.append(float(out.loss))
    return math.exp(sum(losses) / max(1, len(losses)))

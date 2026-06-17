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


def causal_lm_perplexity(model: torch.nn.Module, batches: list[torch.Tensor]) -> float:
    if not batches:
        raise ValueError("Cannot compute perplexity without evaluation batches")
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            out = model(input_ids=batch, labels=batch)
            if out.loss is not None:
                value = float(out.loss)
                if not math.isfinite(value):
                    raise ValueError("Cannot compute perplexity from non-finite loss")
                losses.append(value)
    if not losses:
        raise ValueError("Cannot compute perplexity because model returned no losses")
    try:
        perplexity = math.exp(sum(losses) / len(losses))
    except OverflowError as exc:
        raise ValueError("Computed perplexity is not finite") from exc
    if not math.isfinite(perplexity):
        raise ValueError("Computed perplexity is not finite")
    return perplexity

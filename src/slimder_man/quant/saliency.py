from __future__ import annotations


def composite_saliency(signals: dict[str, float | None], weights: dict[str, float] | None = None, shared: bool = False) -> float:
    weights = weights or {"freq": 0.20, "soft": 0.20, "reap": 0.25, "quant_error": 0.15, "hessian_trace": 0.10, "perplexity_delta": 0.10}
    available = {k: v for k, v in signals.items() if v is not None and k in weights}
    denom = sum(weights[k] for k in available)
    score = 0.0 if denom == 0 else sum((weights[k] / denom) * float(v) for k, v in available.items())
    if shared:
        score += 1.0
    return score

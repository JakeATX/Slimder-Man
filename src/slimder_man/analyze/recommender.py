from __future__ import annotations

from dataclasses import dataclass, asdict

from slimder_man.config.presets import PRESETS


@dataclass
class Candidate:
    preset: str
    hidden_size: int
    remove_last_n_layers: int
    routed_experts: int
    routed_top_k: int
    schedule: str
    estimated_total_params: int
    estimated_active_params: int
    risk: str


def recommend(architecture: dict, preset: str = "balanced_50") -> list[dict]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset}")
    base = PRESETS[preset]
    hidden = int(base["hidden_size"])
    layers = max(1, int(architecture["num_layers"]) - int(base["remove_last_n_layers"]))
    experts = int(base["routed_experts"])
    vocab = int(architecture["vocab_size"])
    est = vocab * hidden * 2 + layers * (hidden * hidden * 2 + experts * hidden * hidden)
    active = vocab * hidden * 2 + layers * (hidden * hidden * 2 + int(base["routed_top_k"]) * hidden * hidden)
    risks = {
        "conservative_20": "low",
        "balanced_50": "medium",
        "slimqwen_anchor": "paper-supported anchor",
        "aggressive_80": "high",
        "extreme_90": "very high",
    }
    candidates = [
        Candidate(preset, hidden, int(base["remove_last_n_layers"]), experts, int(base["routed_top_k"]), str(base["schedule"]), est, active, risks[preset])
    ]
    return [asdict(c) for c in candidates]

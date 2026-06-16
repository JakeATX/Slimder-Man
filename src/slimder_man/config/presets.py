from __future__ import annotations


PRESETS: dict[str, dict[str, int | str]] = {
    "conservative_20": {"hidden_size": 16, "remove_last_n_layers": 0, "routed_experts": 6, "routed_top_k": 2, "schedule": "one_stage"},
    "balanced_50": {"hidden_size": 12, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2, "schedule": "one_stage"},
    "slimqwen_anchor": {"hidden_size": 1536, "remove_last_n_layers": 12, "routed_experts": 256, "routed_top_k": 8, "schedule": "depth_first"},
    "aggressive_80": {"hidden_size": 1280, "remove_last_n_layers": 16, "routed_experts": 128, "routed_top_k": 6, "schedule": "depth_first"},
    "extreme_90": {"hidden_size": 1024, "remove_last_n_layers": 24, "routed_experts": 64, "routed_top_k": 4, "schedule": "depth_first"},
}

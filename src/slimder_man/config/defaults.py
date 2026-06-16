from __future__ import annotations

from .schema import SlimderConfig


def tiny_default_config() -> SlimderConfig:
    return SlimderConfig()


def slimqwen_anchor_config() -> SlimderConfig:
    cfg = SlimderConfig(
        teacher={"model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct", "load_mode": "transformers", "dtype": "bfloat16", "device_map": "auto"},
        project={"name": "qwen3_next_80a3b_to_slimder_23a2b", "output_dir": "runs/slimder_23a2b", "paper_faithful": True},
        calibration={"sample_count": 1024, "sequence_length": 4096},
        compression={
            "preset": "slimqwen_anchor",
            "target": {"hidden_size": 1536, "remove_last_n_layers": 12, "routed_experts": 256, "routed_top_k": 8, "shared_experts": "keep"},
        },
        progressive={"schedule": "depth_first", "stages": 2, "token_split": [0.1, 0.9]},
        training={"token_budget": 400_000_000_000, "global_batch_size": 1024, "sequence_length": 4096, "precision": "bf16"},
    )
    return cfg

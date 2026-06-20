from __future__ import annotations

from pathlib import Path


def write_analysis_report(path: str | Path, architecture: dict, recommendations: list[dict], warnings: list[str] | None = None) -> None:
    lines = ["# Slimder Man Analysis Report", ""]
    if architecture.get("source") == "config_only":
        lines.extend(
            [
                "## Scope",
                "",
                "- source: config_only",
                "- calibration: not_run",
                "- weights_loaded: false",
                "- parameter_counts: not_checkpoint_derived",
                "- recommendation_estimates: formula_based_from_config",
                "- mtp_detection: config_fields_only",
                "",
            ]
        )
    lines.extend(["## Architecture", ""])
    for key in ("model_type", "total_params", "hidden_size", "vocab_size", "num_layers", "has_mtp", "mtp_depths", "tied_embeddings"):
        lines.append(f"- {key}: {architecture.get(key)}")
    lines.extend(["", "## Recommendations", ""])
    for rec in recommendations:
        memory = rec.get("estimated_memory_gib", rec.get("memory_estimate_gb"))
        memory_text = f", memory_gib={memory:.3f}" if isinstance(memory, (int, float)) else ""
        schedule = rec.get("schedule", "unknown")
        lines.append(
            f"- {rec['preset']}: hidden={rec['hidden_size']}, remove_last={rec['remove_last_n_layers']}, "
            f"experts={rec['routed_experts']}, top_k={rec['routed_top_k']}, schedule={schedule}, "
            f"risk={rec['risk']}{memory_text}"
        )
    lines.extend(["", "## Warnings", ""])
    for warning in warnings or []:
        lines.append(f"- {warning}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

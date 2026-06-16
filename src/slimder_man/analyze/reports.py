from __future__ import annotations

from pathlib import Path


def write_analysis_report(path: str | Path, architecture: dict, recommendations: list[dict], warnings: list[str] | None = None) -> None:
    lines = ["# Slimder Man Analysis Report", "", "## Architecture", ""]
    for key in ("model_type", "total_params", "hidden_size", "vocab_size", "num_layers", "has_mtp", "mtp_depths", "tied_embeddings"):
        lines.append(f"- {key}: {architecture.get(key)}")
    lines.extend(["", "## Recommendations", ""])
    for rec in recommendations:
        lines.append(
            f"- {rec['preset']}: hidden={rec['hidden_size']}, remove_last={rec['remove_last_n_layers']}, "
            f"experts={rec['routed_experts']}, top_k={rec['routed_top_k']}, risk={rec['risk']}"
        )
    lines.extend(["", "## Warnings", ""])
    for warning in warnings or []:
        lines.append(f"- {warning}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

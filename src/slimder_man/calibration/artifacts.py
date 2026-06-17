from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from slimder_man.calibration.collectors import CalibrationResult, hidden_keep_indices
from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import sha256_file
from slimder_man.utils.json import write_json


SIMILARITY_ATTRS = {
    "router_logits": "router_logits_similarity",
    "router_weights": "router_weights_similarity",
    "expert_outputs": "expert_outputs_similarity",
}


def _cpu_tensor(value: torch.Tensor, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
    tensor = value.detach().cpu()
    return tensor.to(dtype) if dtype is not None and tensor.is_floating_point() else tensor


def _similarity_by_metric(calibration: CalibrationResult, metric: str) -> list[torch.Tensor]:
    attr = SIMILARITY_ATTRS[metric]
    values = getattr(calibration, attr, None)
    if values is None:
        if metric == "router_weights":
            return calibration.expert_similarity
        return []
    return values


def _record_artifact(out_dir: Path, artifacts: dict[str, dict[str, Any]], path: Path, kind: str) -> None:
    artifacts[path.name] = {
        "path": path.name,
        "kind": kind,
        "sha256": sha256_file(path),
    }


def write_calibration_artifacts(
    out_dir: str | Path,
    cfg: SlimderConfig,
    calibration: CalibrationResult,
    calibration_source_manifest: dict[str, Any],
    architecture: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist calibration tensors and deterministic provenance for analysis.

    The artifact names are intentionally stable because downstream compression
    stages and external audit tools can refer to them without importing Python.
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_hidden_size = cfg.compression.target.hidden_size
    keep_idx = hidden_keep_indices(calibration.hidden_scores, target_hidden_size)
    artifacts: dict[str, dict[str, Any]] = {}

    hidden_path = out_path / "hidden_importance.safetensors"
    hidden_tensors = {"global": _cpu_tensor(calibration.hidden_scores)}
    hidden_tensors.update(
        {f"layer_{idx}": _cpu_tensor(scores) for idx, scores in enumerate(calibration.per_layer_hidden_scores)}
    )
    save_file(hidden_tensors, hidden_path)
    _record_artifact(out_path, artifacts, hidden_path, "hidden_importance")

    keep_path = out_path / f"hidden_keep_indices_{target_hidden_size}.json"
    write_json(
        keep_path,
        {
            "target_hidden_size": target_hidden_size,
            "source_hidden_size": int(calibration.hidden_scores.numel()),
            "method": cfg.compression.width.method,
            "indices": [int(x) for x in keep_idx.tolist()],
        },
    )
    _record_artifact(out_path, artifacts, keep_path, "hidden_keep_indices")

    layer_summaries = []
    for layer_idx, (freq, soft, reap) in enumerate(
        zip(calibration.expert_frequency, calibration.expert_soft, calibration.expert_reap, strict=True)
    ):
        stats_path = out_path / f"expert_stats_layer_{layer_idx}.safetensors"
        save_file(
            {
                "frequency": _cpu_tensor(freq),
                "soft_logits": _cpu_tensor(soft),
                "reap": _cpu_tensor(reap),
            },
            stats_path,
        )
        _record_artifact(out_path, artifacts, stats_path, "expert_stats")

        similarity_artifacts: dict[str, str] = {}
        for metric in SIMILARITY_ATTRS:
            matrices = _similarity_by_metric(calibration, metric)
            if layer_idx >= len(matrices):
                continue
            sim_path = out_path / f"expert_similarity_layer_{layer_idx}_{metric}.safetensors"
            save_file({"similarity": _cpu_tensor(matrices[layer_idx])}, sim_path)
            _record_artifact(out_path, artifacts, sim_path, f"expert_similarity:{metric}")
            similarity_artifacts[metric] = sim_path.name

        top_frequency = torch.argsort(freq.detach().cpu(), descending=True, stable=True).tolist()
        layer_summaries.append(
            {
                "layer_idx": layer_idx,
                "num_experts": int(freq.numel()),
                "importance_artifact": stats_path.name,
                "similarity_artifacts": similarity_artifacts,
                "top_experts_by_frequency": [int(x) for x in top_frequency],
            }
        )

    routing_summary = {
        "representation": calibration.representation,
        "importance_metric": cfg.compression.experts.importance_metric,
        "similarity_metric": cfg.compression.experts.similarity_metric,
        "layers": layer_summaries,
    }
    routing_path = out_path / "routing_summary.json"
    write_json(routing_path, routing_summary)
    _record_artifact(out_path, artifacts, routing_path, "routing_summary")

    manifest = {
        "schema_version": "1.0",
        "teacher_model": cfg.teacher.model_id_or_path,
        "teacher_revision": cfg.teacher.revision,
        "teacher_load_mode": cfg.teacher.load_mode,
        "seed": cfg.calibration.seed,
        "project_seed": cfg.project.seed,
        "target": cfg.compression.target.model_dump(mode="json"),
        "width": {
            "method": cfg.compression.width.method,
            "hidden_size_before": int(calibration.hidden_scores.numel()),
            "hidden_size_after": target_hidden_size,
            "keep_indices_artifact": keep_path.name,
        },
        "experts": {
            "importance_metric": cfg.compression.experts.importance_metric,
            "similarity_metric": cfg.compression.experts.similarity_metric,
            "available_similarity_metrics": [
                metric for metric in SIMILARITY_ATTRS if len(_similarity_by_metric(calibration, metric)) > 0
            ],
        },
        "calibration": calibration_source_manifest,
        "architecture": architecture or {},
        "artifacts": artifacts,
    }
    manifest_path = out_path / "calibration_manifest.json"
    write_json(manifest_path, manifest)
    return manifest

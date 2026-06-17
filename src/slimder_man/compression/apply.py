from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import inspect
from types import SimpleNamespace

import torch

from slimder_man.calibration.artifacts import SIMILARITY_ATTRS
from slimder_man.adapters.registry import get_adapter
from slimder_man.adapters.tiny import TinyAdapter, TinyMoEForCausalLM, clone_tiny_model
from slimder_man.calibration.collectors import CalibrationResult, hidden_keep_indices
from slimder_man.compression.depth import compute_depth_keep_indices, resolve_remove_last_n
from slimder_man.compression.experts import merge_experts
from slimder_man.compression.manifests import save_manifest
from slimder_man.compression.router import router_rows_for_merge
from slimder_man.compression.validate import validate_tiny_model
from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import sha256_file
from slimder_man.utils.json import read_json
from slimder_man.utils.provenance import config_provenance


MODEL_ARTIFACT_NAMES = {"config.json", "generation_config.json", "model.safetensors", "pytorch_model.bin"}
TOKENIZER_ARTIFACT_NAMES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "added_tokens.json",
}


def _relative_file(path: Path, root: Path) -> str | None:
    try:
        resolved = path.resolve()
        root_resolved = root.resolve()
        if not resolved.is_file() or not resolved.is_relative_to(root_resolved):
            return None
        return resolved.relative_to(root_resolved).as_posix()
    except OSError:
        return None


def _flatten_saved_paths(value) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [Path(value)]
    if isinstance(value, dict):
        paths: list[Path] = []
        for item in value.values():
            paths.extend(_flatten_saved_paths(item))
        return paths
    if isinstance(value, (list, tuple, set)):
        paths = []
        for item in value:
            paths.extend(_flatten_saved_paths(item))
        return paths
    return []


def _hash_model_artifacts(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if (
            name in MODEL_ARTIFACT_NAMES
            or name.startswith("model-") and name.endswith(".safetensors")
            or name.startswith("pytorch_model-") and name.endswith(".bin")
            or name.endswith(".safetensors.index.json")
            or name.endswith(".bin.index.json")
        ):
            rel = path.relative_to(output_dir).as_posix()
            hashes[rel] = sha256_file(path)
    return hashes


def _has_safetensors_checkpoint(output_dir: Path) -> bool:
    return (
        (output_dir / "model.safetensors").exists()
        or any(output_dir.glob("model-*.safetensors"))
        or (output_dir / "model.safetensors.index.json").exists()
    )


def _has_torch_checkpoint(output_dir: Path) -> bool:
    return (
        (output_dir / "pytorch_model.bin").exists()
        or any(output_dir.glob("pytorch_model-*.bin"))
        or (output_dir / "pytorch_model.bin.index.json").exists()
    )


def _save_transformers_checkpoint(model, adapter, output_dir: Path, manifest: dict, output_format: str) -> None:
    safe_serialization = output_format == "hf_safetensors"
    if hasattr(model, "save_pretrained"):
        save_pretrained = model.save_pretrained
        signature = inspect.signature(save_pretrained)
        accepts_safe_serialization = "safe_serialization" in signature.parameters or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        )
        if accepts_safe_serialization:
            model.save_pretrained(output_dir, safe_serialization=safe_serialization)
        else:
            model.save_pretrained(output_dir)
    else:
        adapter.save_pretrained(model, str(output_dir), manifest)
    if output_format == "hf_safetensors" and not _has_safetensors_checkpoint(output_dir):
        raise ValueError("student.output_format=hf_safetensors did not produce a safetensors checkpoint")
    if output_format == "torch" and not _has_torch_checkpoint(output_dir):
        raise ValueError("student.output_format=torch did not produce a PyTorch checkpoint")


def _save_tokenizer_artifacts(tokenizer, output_dir: Path) -> dict[str, str]:
    if tokenizer is None:
        return {}
    if not hasattr(tokenizer, "save_pretrained"):
        raise ValueError("Tokenizer object does not expose save_pretrained")
    config_path = output_dir / "config.json"
    config_hash_before = sha256_file(config_path) if config_path.exists() else None
    saved = tokenizer.save_pretrained(output_dir)
    if config_hash_before is not None:
        if not config_path.exists():
            raise ValueError("Tokenizer save_pretrained removed model config.json")
        if sha256_file(config_path) != config_hash_before:
            raise ValueError("Tokenizer save_pretrained modified model config.json")
    candidates = set()
    for saved_path in _flatten_saved_paths(saved):
        rel = _relative_file(saved_path, output_dir)
        if rel is not None:
            candidates.add(rel)
    for path in output_dir.rglob("*"):
        if path.is_file() and path.name in TOKENIZER_ARTIFACT_NAMES:
            candidates.add(path.relative_to(output_dir).as_posix())
    hashes: dict[str, str] = {}
    for name in sorted(candidates):
        path = output_dir / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def _calibration_manifest_reference(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ValueError(f"Calibration manifest not found: {manifest_path}")
    manifest_path = manifest_path.resolve()
    manifest = read_json(manifest_path)
    analysis_dir = manifest_path.parent
    artifacts = manifest.get("artifacts", {})
    verified: dict[str, dict] = {}
    for name, artifact in artifacts.items():
        artifact_path = analysis_dir / artifact["path"]
        if not artifact_path.exists():
            raise ValueError(f"Calibration artifact missing: {artifact_path}")
        digest = sha256_file(artifact_path)
        if digest != artifact.get("sha256"):
            raise ValueError(f"Calibration artifact hash mismatch: {artifact_path}")
        verified[name] = {
            "path": str(artifact_path.resolve()),
            "kind": artifact.get("kind"),
            "sha256": digest,
        }
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "analysis_dir": str(analysis_dir.resolve()),
        "artifacts": verified,
        "calibration": manifest.get("calibration"),
        "reap_convention": manifest.get("experts", {}).get("reap_convention"),
    }


def _layer_calibration_references(calibration_ref: dict | None, layer_idx: int, importance_metric: str, similarity_metric: str) -> dict:
    if calibration_ref is None:
        return {}
    artifacts = calibration_ref.get("artifacts", {})
    score_name = f"expert_stats_layer_{layer_idx}.safetensors"
    refs: dict[str, dict] = {}
    if score_name in artifacts:
        refs["score_artifact"] = {
            **artifacts[score_name],
            "tensor": importance_metric,
        }
    if similarity_metric in SIMILARITY_ATTRS:
        sim_name = f"expert_similarity_layer_{layer_idx}_{similarity_metric}.safetensors"
        if sim_name in artifacts:
            refs["similarity_artifact"] = {
                **artifacts[sim_name],
                "tensor": "similarity",
                "metric": similarity_metric,
            }
    return refs


def _resolve_expert_importance(calibration: CalibrationResult, metric: str, layer_idx: int) -> torch.Tensor:
    metrics = {
        "frequency": calibration.expert_frequency,
        "soft_logits": calibration.expert_soft,
        "reap": calibration.expert_reap,
    }
    if metric not in metrics:
        raise ValueError(f"Unsupported expert importance metric {metric}")
    return metrics[metric][layer_idx]


def _resolve_expert_similarity(calibration: CalibrationResult, metric: str, layer_idx: int) -> torch.Tensor:
    attr_by_metric = {
        "router_weights": "router_weights_similarity",
        "router_logits": "router_logits_similarity",
        "expert_outputs": "expert_outputs_similarity",
    }
    if metric not in attr_by_metric:
        raise ValueError(f"Unsupported expert similarity metric {metric}")
    attr = attr_by_metric[metric]
    value = getattr(calibration, attr, None)
    if value is None:
        if metric == "router_weights":
            return calibration.expert_similarity[layer_idx]
        raise ValueError(f"Calibration result does not include {metric} similarity")
    return value[layer_idx]


def _prune_experts(experts: list[torch.nn.Module], scores: torch.Tensor, target_experts: int):
    if target_experts <= 0 or target_experts > len(experts):
        raise ValueError("target expert count must be between 1 and original expert count")
    order = torch.argsort(scores, descending=True, stable=True).tolist()[:target_experts]
    plan = SimpleNamespace(s_keep=order, s_base=[], groups={}, new_expert_order=order, warning=None)
    return [deepcopy(experts[i]) for i in order], plan


def _select_or_merge_experts(experts: list[torch.nn.Module], scores: torch.Tensor, sim: torch.Tensor, target_experts: int, method: str):
    if target_experts == len(experts):
        order = list(range(len(experts)))
        plan = SimpleNamespace(s_keep=order, s_base=[], groups={}, new_expert_order=order, warning=None)
        return [deepcopy(expert) for expert in experts], plan
    if method == "prune":
        return _prune_experts(experts, scores, target_experts)
    return merge_experts(experts, scores, sim, target_experts)


def compress_tiny_model(
    model: TinyMoEForCausalLM,
    cfg: SlimderConfig,
    calibration: CalibrationResult,
    output_dir: str | Path | None = None,
    calibration_manifest_path: str | Path | None = None,
    source_config_path: str | Path | None = None,
    stage_provenance: dict | None = None,
) -> tuple[TinyMoEForCausalLM, dict]:
    student = clone_tiny_model(model)
    adapter = TinyAdapter()
    target = cfg.compression.target
    calibration_ref = _calibration_manifest_reference(calibration_manifest_path)
    remove_last_n_layers = resolve_remove_last_n(len(student.layers), target.remove_last_n_layers, target.depth_remove_fraction)
    keep_blocks = compute_depth_keep_indices(len(student.layers), remove_last_n_layers)
    adapter.drop_blocks(student, keep_blocks)
    keep_idx = hidden_keep_indices(calibration.hidden_scores, target.hidden_size)
    adapter.slice_hidden_channels(student, keep_idx)
    expert_layers = []
    for layer_idx, block in enumerate(student.layers):
        original_layer_idx = keep_blocks[layer_idx]
        moe = block.moe
        scores = _resolve_expert_importance(calibration, cfg.compression.experts.importance_metric, original_layer_idx)
        sim = _resolve_expert_similarity(calibration, cfg.compression.experts.similarity_metric, original_layer_idx)
        old_router_rows = moe.router.weight.detach().clone()
        new_experts, plan = _select_or_merge_experts(list(moe.experts), scores, sim, target.routed_experts, cfg.compression.experts.method)
        rows = router_rows_for_merge(old_router_rows, plan.s_keep, plan.s_base, cfg.compression.experts.router_row_strategy)
        adapter.replace_experts(moe, new_experts, rows, target.routed_top_k)
        expert_layers.append(
            {
                "layer_idx": original_layer_idx,
                "s_keep": plan.s_keep,
                "s_base": plan.s_base,
                "groups": {str(k): v for k, v in plan.groups.items()},
                "new_expert_order": plan.new_expert_order,
                "warning": plan.warning,
                "score_vector": [float(x) for x in scores.detach().cpu().tolist()],
                "importance_metric_used": cfg.compression.experts.importance_metric,
                "similarity_metric_used": cfg.compression.experts.similarity_metric,
                **_layer_calibration_references(
                    calibration_ref,
                    original_layer_idx,
                    cfg.compression.experts.importance_metric,
                    cfg.compression.experts.similarity_metric,
                ),
            }
        )
    manifest = {
        "schema_version": "1.0",
        "paper_faithful": cfg.project.paper_faithful,
        "teacher_model": cfg.teacher.model_id_or_path,
        "teacher_revision": cfg.teacher.revision,
        "student_output_format": "torch",
        "seed": cfg.project.seed,
        "provenance": config_provenance(cfg, source_config_path),
        "calibration_artifacts": calibration_ref,
        "progressive": cfg.progressive.model_dump(mode="json"),
        "stage_provenance": stage_provenance or {},
        "calibration": {"sample_count": cfg.calibration.sample_count, "sequence_length": cfg.calibration.sequence_length},
        "target": {
            "hidden_size": target.hidden_size,
            "remove_last_n_layers": remove_last_n_layers,
            "routed_experts": target.routed_experts,
            "top_k": target.routed_top_k,
        },
        "depth": {"method": "last_layers", "kept_block_indices": keep_blocks},
        "width": {
            "method": "rmsnorm_mean_abs",
            "hidden_keep_indices": keep_idx.tolist(),
            "hidden_size_before": model.config.hidden_size,
            "hidden_size_after": target.hidden_size,
        },
        "experts": {
            "method": cfg.compression.experts.method,
            "importance_metric": cfg.compression.experts.importance_metric,
            "similarity_metric": cfg.compression.experts.similarity_metric,
            "layers": expert_layers,
        },
        "router": {"row_strategy": "base", "top_k_before": model.config.top_k, "top_k_after": target.routed_top_k},
        "param_counts": {"before": sum(p.numel() for p in model.parameters()), "after": sum(p.numel() for p in student.parameters()), "actual_after": sum(p.numel() for p in student.parameters())},
    }
    adapter.update_config_after_compression(student, manifest)
    errors = validate_tiny_model(student)
    if errors:
        raise ValueError("; ".join(errors))
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        adapter.save_pretrained(student, str(output_dir), manifest)
        hashes = {}
        for filename in ("model.pt", "config.json"):
            path = Path(output_dir) / filename
            if path.exists():
                hashes[filename] = sha256_file(path)
        manifest["artifact_hashes"] = hashes
        save_manifest(Path(output_dir) / "compression_manifest.json", manifest)
    return student, manifest


def compress_model(
    model,
    cfg: SlimderConfig,
    calibration: CalibrationResult,
    adapter=None,
    output_dir: str | Path | None = None,
    tokenizer=None,
    calibration_manifest_path: str | Path | None = None,
    source_config_path: str | Path | None = None,
    stage_provenance: dict | None = None,
):
    if isinstance(model, TinyMoEForCausalLM):
        return compress_tiny_model(
            model,
            cfg,
            calibration,
            output_dir,
            calibration_manifest_path=calibration_manifest_path,
            source_config_path=source_config_path,
            stage_provenance=stage_provenance,
        )

    student = deepcopy(model)
    adapter = adapter or get_adapter(student)
    target = cfg.compression.target
    calibration_ref = _calibration_manifest_reference(calibration_manifest_path)
    source_layers = len(adapter.iter_transformer_blocks(student))
    remove_last_n_layers = resolve_remove_last_n(source_layers, target.remove_last_n_layers, target.depth_remove_fraction)
    keep_blocks = compute_depth_keep_indices(source_layers, remove_last_n_layers)
    adapter.drop_blocks(student, keep_blocks)

    current_hidden = adapter.describe_architecture(student).hidden_size
    keep_idx = list(range(current_hidden))
    if target.hidden_size != current_hidden:
        keep_tensor = hidden_keep_indices(calibration.hidden_scores, target.hidden_size)
        adapter.slice_hidden_channels(student, keep_tensor)
        keep_idx = keep_tensor.tolist()

    expert_layers = []
    for layer_idx, moe in enumerate(adapter.iter_moe_layers(student)):
        original_layer_idx = keep_blocks[layer_idx] if layer_idx < len(keep_blocks) else layer_idx
        scores = _resolve_expert_importance(calibration, cfg.compression.experts.importance_metric, original_layer_idx)
        sim = _resolve_expert_similarity(calibration, cfg.compression.experts.similarity_metric, original_layer_idx)
        experts = adapter.get_routed_experts(moe)
        old_router_rows = adapter.get_router(moe).weight.detach().clone()
        new_experts, plan = _select_or_merge_experts(experts, scores, sim, target.routed_experts, cfg.compression.experts.method)
        rows = router_rows_for_merge(old_router_rows, plan.s_keep, plan.s_base, cfg.compression.experts.router_row_strategy)
        adapter.replace_experts(moe, new_experts, rows, target.routed_top_k)
        expert_layers.append(
            {
                "layer_idx": original_layer_idx,
                "s_keep": plan.s_keep,
                "s_base": plan.s_base,
                "groups": {str(k): v for k, v in plan.groups.items()},
                "new_expert_order": plan.new_expert_order,
                "warning": plan.warning,
                "score_vector": [float(x) for x in scores.detach().cpu().tolist()],
                "importance_metric_used": cfg.compression.experts.importance_metric,
                "similarity_metric_used": cfg.compression.experts.similarity_metric,
                **_layer_calibration_references(
                    calibration_ref,
                    original_layer_idx,
                    cfg.compression.experts.importance_metric,
                    cfg.compression.experts.similarity_metric,
                ),
            }
        )

    arch = adapter.describe_architecture(student)
    manifest = {
        "schema_version": "1.0",
        "paper_faithful": cfg.project.paper_faithful,
        "teacher_model": cfg.teacher.model_id_or_path,
        "teacher_revision": cfg.teacher.revision,
        "student_output_format": cfg.student.output_format,
        "seed": cfg.project.seed,
        "provenance": config_provenance(cfg, source_config_path),
        "calibration_artifacts": calibration_ref,
        "progressive": cfg.progressive.model_dump(mode="json"),
        "stage_provenance": stage_provenance or {},
        "calibration": {"sample_count": cfg.calibration.sample_count, "sequence_length": cfg.calibration.sequence_length},
        "target": {"hidden_size": target.hidden_size, "remove_last_n_layers": remove_last_n_layers, "routed_experts": target.routed_experts, "top_k": target.routed_top_k},
        "depth": {"method": "last_layers", "kept_block_indices": keep_blocks},
        "width": {"method": "rmsnorm_mean_abs", "hidden_keep_indices": keep_idx, "hidden_size_before": current_hidden, "hidden_size_after": target.hidden_size},
        "experts": {"method": cfg.compression.experts.method, "importance_metric": cfg.compression.experts.importance_metric, "similarity_metric": cfg.compression.experts.similarity_metric, "layers": expert_layers},
        "router": {"row_strategy": cfg.compression.experts.router_row_strategy, "top_k_before": getattr(getattr(model, "config", None), "num_experts_per_tok", target.routed_top_k), "top_k_after": target.routed_top_k},
        "param_counts": {"before": sum(p.numel() for p in model.parameters()), "after": sum(p.numel() for p in student.parameters()), "actual_after": sum(p.numel() for p in student.parameters())},
    }
    adapter.update_config_after_compression(student, manifest)
    input_ids = torch.randint(0, arch.vocab_size, (1, min(8, max(2, arch.vocab_size))))
    with torch.no_grad():
        out = student(input_ids=input_ids, labels=input_ids)
    if not torch.isfinite(out.logits).all() or (out.loss is not None and not torch.isfinite(out.loss)):
        raise ValueError("Compressed model failed finite forward validation")
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        _save_transformers_checkpoint(student, adapter, out_path, manifest, cfg.student.output_format)
        tokenizer_hashes = _save_tokenizer_artifacts(tokenizer, out_path)
        hashes = _hash_model_artifacts(out_path)
        hashes.update(tokenizer_hashes)
        manifest["artifact_hashes"] = hashes
        if tokenizer_hashes:
            manifest["tokenizer"] = {"saved": True, "artifact_hashes": tokenizer_hashes}
        else:
            manifest["tokenizer"] = {"saved": False}
        save_manifest(out_path / "compression_manifest.json", manifest)
    return student, manifest

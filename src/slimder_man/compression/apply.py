from __future__ import annotations

from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

import torch

from slimder_man.adapters.registry import get_adapter
from slimder_man.adapters.tiny import TinyAdapter, TinyMoEForCausalLM, clone_tiny_model
from slimder_man.calibration.collectors import CalibrationResult, hidden_keep_indices
from slimder_man.compression.depth import compute_depth_keep_indices
from slimder_man.compression.experts import merge_experts
from slimder_man.compression.manifests import save_manifest
from slimder_man.compression.router import router_rows_for_merge
from slimder_man.compression.validate import validate_tiny_model
from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import sha256_file


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


def compress_tiny_model(model: TinyMoEForCausalLM, cfg: SlimderConfig, calibration: CalibrationResult, output_dir: str | Path | None = None) -> tuple[TinyMoEForCausalLM, dict]:
    student = clone_tiny_model(model)
    adapter = TinyAdapter()
    target = cfg.compression.target
    keep_blocks = compute_depth_keep_indices(len(student.layers), target.remove_last_n_layers)
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
            }
        )
    manifest = {
        "schema_version": "1.0",
        "paper_faithful": cfg.project.paper_faithful,
        "teacher_model": cfg.teacher.model_id_or_path,
        "teacher_revision": cfg.teacher.revision,
        "seed": cfg.project.seed,
        "calibration": {"sample_count": cfg.calibration.sample_count, "sequence_length": cfg.calibration.sequence_length},
        "target": {
            "hidden_size": target.hidden_size,
            "remove_last_n_layers": target.remove_last_n_layers,
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


def compress_model(model, cfg: SlimderConfig, calibration: CalibrationResult, adapter=None, output_dir: str | Path | None = None):
    if isinstance(model, TinyMoEForCausalLM):
        return compress_tiny_model(model, cfg, calibration, output_dir)

    student = deepcopy(model)
    adapter = adapter or get_adapter(student)
    target = cfg.compression.target
    keep_blocks = compute_depth_keep_indices(len(adapter.iter_transformer_blocks(student)), target.remove_last_n_layers)
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
            }
        )

    arch = adapter.describe_architecture(student)
    manifest = {
        "schema_version": "1.0",
        "paper_faithful": cfg.project.paper_faithful,
        "teacher_model": cfg.teacher.model_id_or_path,
        "teacher_revision": cfg.teacher.revision,
        "seed": cfg.project.seed,
        "calibration": {"sample_count": cfg.calibration.sample_count, "sequence_length": cfg.calibration.sequence_length},
        "target": {"hidden_size": target.hidden_size, "remove_last_n_layers": target.remove_last_n_layers, "routed_experts": target.routed_experts, "top_k": target.routed_top_k},
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
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        adapter.save_pretrained(student, str(output_dir), manifest)
        hashes = {}
        for filename in ("model.safetensors", "pytorch_model.bin", "config.json"):
            path = Path(output_dir) / filename
            if path.exists():
                hashes[filename] = sha256_file(path)
        manifest["artifact_hashes"] = hashes
        save_manifest(Path(output_dir) / "compression_manifest.json", manifest)
    return student, manifest

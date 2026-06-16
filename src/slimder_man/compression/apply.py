from __future__ import annotations

from pathlib import Path

import torch

from slimder_man.adapters.tiny import TinyAdapter, TinyMoEForCausalLM, clone_tiny_model
from slimder_man.calibration.collectors import CalibrationResult, hidden_keep_indices
from slimder_man.compression.depth import compute_depth_keep_indices
from slimder_man.compression.experts import merge_experts
from slimder_man.compression.manifests import save_manifest
from slimder_man.compression.router import router_rows_for_merge
from slimder_man.compression.validate import validate_tiny_model
from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import sha256_file


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
        scores = calibration.expert_soft[original_layer_idx]
        sim = calibration.expert_similarity[original_layer_idx]
        old_router_rows = moe.router.weight.detach().clone()
        new_experts, plan = merge_experts(list(moe.experts), scores, sim, target.routed_experts)
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

from __future__ import annotations

import hashlib
import json
from typing import Any

from slimder_man.analyze.recommender import recommend
from slimder_man.config.schema import CompressionPlanConfig, SlimderConfig


def architecture_fingerprint(architecture: dict[str, Any]) -> str:
    payload = {
        "model_type": architecture.get("model_type"),
        "hidden_size": architecture.get("hidden_size"),
        "vocab_size": architecture.get("vocab_size"),
        "num_layers": architecture.get("num_layers"),
        "block_kinds": architecture.get("block_kinds"),
        "moe_layers": _normalized_moe_layers(architecture.get("moe_layers") or []),
        "has_mtp": architecture.get("has_mtp"),
        "mtp_depths": architecture.get("mtp_depths"),
        "tied_embeddings": architecture.get("tied_embeddings"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def apply_recommendation_to_config(
    cfg: SlimderConfig,
    architecture: dict[str, Any],
    preset: str | None = None,
    candidate_id: str | None = None,
) -> tuple[SlimderConfig, dict[str, Any]]:
    preset_name = preset or cfg.compression.preset
    candidates = recommend(architecture, preset_name)
    candidate = _select_candidate(candidates, candidate_id)
    cfg.compression.preset = preset_name
    cfg.compression.target.hidden_size = int(candidate["hidden_size"])
    cfg.compression.target.remove_last_n_layers = int(candidate["remove_last_n_layers"])
    cfg.compression.target.routed_experts = int(candidate["routed_experts"])
    cfg.compression.target.routed_top_k = int(candidate["routed_top_k"])
    cfg.compression.plan = CompressionPlanConfig(
        candidate_id=str(candidate["candidate_id"]),
        preset=preset_name,
        source_architecture_fingerprint=architecture_fingerprint(architecture),
        source_summary=_source_summary(architecture),
        target=cfg.compression.target.model_copy(deep=True),
    )
    return cfg, compression_plan_payload(cfg, architecture, candidate)


def compression_plan_payload(cfg: SlimderConfig, architecture: dict[str, Any], candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    if cfg.compression.plan is None:
        return {}
    payload = cfg.compression.plan.model_dump(mode="json")
    payload["current_architecture_fingerprint"] = architecture_fingerprint(architecture)
    if candidate is not None:
        payload["candidate"] = candidate
    return payload


def validate_applied_plan(cfg: SlimderConfig, architecture: dict[str, Any], *, required: bool = False) -> None:
    plan = cfg.compression.plan
    if plan is None:
        if required:
            raise ValueError(
                "compression.plan is required before compressing this Transformers checkpoint. "
                "Run slimder recommend --candidate-id <id> --write-config <path> first."
            )
        return
    actual = architecture_fingerprint(architecture)
    if actual != plan.source_architecture_fingerprint:
        raise ValueError(
            "compression.plan source architecture fingerprint mismatch: "
            f"config={plan.source_architecture_fingerprint}, current={actual}. "
            "Re-run slimder recommend --candidate-id ... --write-config ... for this teacher checkpoint."
        )
    target = cfg.compression.target
    mismatches = []
    for field in ("hidden_size", "remove_last_n_layers", "routed_experts", "routed_top_k", "shared_experts"):
        if getattr(target, field) != getattr(plan.target, field):
            mismatches.append(f"{field}: target={getattr(target, field)!r}, plan={getattr(plan.target, field)!r}")
    if mismatches:
        raise ValueError("compression.target does not match compression.plan target; " + "; ".join(mismatches))


def _select_candidate(candidates: list[dict[str, Any]], candidate_id: str | None) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No compression candidates were produced for this architecture")
    if candidate_id is None:
        return candidates[0]
    for candidate in candidates:
        if candidate["candidate_id"] == candidate_id:
            return candidate
    valid = ", ".join(str(candidate["candidate_id"]) for candidate in candidates)
    raise ValueError(f"Unknown candidate_id {candidate_id!r}; valid candidates: {valid}")


def _source_summary(architecture: dict[str, Any]) -> dict[str, Any]:
    moe_layers = architecture.get("moe_layers") or []
    first_moe = moe_layers[0] if moe_layers else {}
    return {
        "model_type": architecture.get("model_type"),
        "hidden_size": architecture.get("hidden_size"),
        "num_layers": architecture.get("num_layers"),
        "vocab_size": architecture.get("vocab_size"),
        "num_moe_layers": len(moe_layers),
        "routed_experts": first_moe.get("num_routed_experts"),
        "shared_experts": first_moe.get("num_shared_experts"),
        "top_k": first_moe.get("top_k"),
    }


def _normalized_moe_layers(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for layer in layers:
        normalized.append(
            {
                "layer_idx": layer.get("layer_idx"),
                "num_routed_experts": layer.get("num_routed_experts"),
                "num_shared_experts": layer.get("num_shared_experts"),
                "top_k": layer.get("top_k"),
            }
        )
    return normalized

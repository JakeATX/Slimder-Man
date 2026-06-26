from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from slimder_man.config.schema import SlimderConfig


@dataclass(frozen=True)
class ModelProfile:
    model_id: str
    family: str
    total_params_b: float
    active_params_b: float | None
    hidden_size: int
    layers: int
    routed_experts: int | None = None
    top_k: int | None = None
    shared_experts: int | None = None
    native_context: int | None = None
    notes: tuple[str, ...] = ()


QWEN36_35B_A3B = ModelProfile(
    model_id="Qwen/Qwen3.6-35B-A3B",
    family="qwen3.6-a3b",
    total_params_b=35.0,
    active_params_b=3.0,
    hidden_size=2048,
    layers=40,
    routed_experts=256,
    top_k=8,
    shared_experts=1,
    native_context=262_144,
    notes=(
        "Model card: 35B total parameters, 3B activated parameters, 40 layers, hidden size 2048.",
        "MoE profile: 256 routed experts, 8 routed experts active per token, plus 1 shared expert.",
        "Native context is 262K tokens; long-context runs require substantial KV/cache and activation headroom.",
    ),
)

QWEN35_35B_A3B = ModelProfile(
    model_id="Qwen/Qwen3.5-35B-A3B",
    family="qwen3.5-a3b",
    total_params_b=35.0,
    active_params_b=3.0,
    hidden_size=2048,
    layers=40,
    routed_experts=256,
    top_k=8,
    shared_experts=1,
    native_context=262_144,
    notes=("Same sizing guidance as Qwen3.6-35B-A3B unless the loaded config proves otherwise.",),
)

QWEN3_NEXT_80B_A3B = ModelProfile(
    model_id="Qwen/Qwen3-Next-80B-A3B-Instruct",
    family="qwen3-next-a3b",
    total_params_b=80.0,
    active_params_b=3.0,
    hidden_size=2048,
    layers=48,
    routed_experts=512,
    top_k=10,
    shared_experts=1,
    native_context=262_144,
    notes=("SlimQwen anchor profile: 80B-class teacher with roughly 3B active parameters.",),
)

KNOWN_PROFILES = {
    "qwen/qwen3.6-35b-a3b": QWEN36_35B_A3B,
    "qwen3.6-35b-a3b": QWEN36_35B_A3B,
    "qwen/qwen3.5-35b-a3b": QWEN35_35B_A3B,
    "qwen3.5-35b-a3b": QWEN35_35B_A3B,
    "qwen/qwen3-next-80b-a3b-instruct": QWEN3_NEXT_80B_A3B,
    "qwen3-next-80b-a3b-instruct": QWEN3_NEXT_80B_A3B,
}


def compute_guidance(cfg: SlimderConfig, architecture: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = _profile_from_config(cfg, architecture)
    weight_memory = _weight_memory(profile.total_params_b)
    target = cfg.compression.target
    target_total_b = _estimate_target_total_params_b(profile, cfg)
    target_weight_memory = _weight_memory(target_total_b)
    long_context = min(cfg.training.sequence_length, profile.native_context or cfg.training.sequence_length)
    local_fit = _local_fit_notes(profile, cfg, weight_memory)
    api_guidance = _api_guidance(cfg)
    remote = _remote_guidance(profile, cfg, weight_memory, target_weight_memory)
    return {
        "model": {
            "model_id": profile.model_id,
            "family": profile.family,
            "source": "known_profile" if _known_profile(cfg.teacher.model_id_or_path) else "architecture_or_config",
            "total_params_b": profile.total_params_b,
            "active_params_b": profile.active_params_b,
            "hidden_size": profile.hidden_size,
            "layers": profile.layers,
            "routed_experts": profile.routed_experts,
            "top_k": profile.top_k,
            "shared_experts": profile.shared_experts,
            "native_context": profile.native_context,
            "notes": list(profile.notes),
        },
        "memory_estimates_gb": {
            "teacher_weights_fp16_or_bf16": round(weight_memory["fp16"], 1),
            "teacher_weights_fp32": round(weight_memory["fp32"], 1),
            "target_student_weights_fp16_or_bf16": round(target_weight_memory["fp16"], 1),
            "minimum_teacher_inference_headroom": round(weight_memory["fp16"] * 1.25, 1),
            "teacher_plus_student_training_floor": round((weight_memory["fp16"] + target_weight_memory["fp16"]) * 2.2, 1),
        },
        "requested_run": {
            "backend": cfg.runtime.backend,
            "sequence_length": cfg.training.sequence_length,
            "long_context_warning": cfg.training.sequence_length > 8192,
            "context_considered": long_context,
            "paper_faithful": cfg.project.paper_faithful,
            "kd_teacher_mode": cfg.kd.teacher_mode,
        },
        "local": local_fit,
        "remote": remote,
        "api": api_guidance,
        "recommended_path": _recommended_path(cfg, profile),
        "next_steps": _next_steps(cfg),
    }


def compute_guidance_markdown(guidance: dict[str, Any]) -> str:
    model = guidance["model"]
    mem = guidance["memory_estimates_gb"]
    requested = guidance["requested_run"]
    remote = guidance["remote"]
    api = guidance["api"]
    lines = [
        f"# Compute guidance for {model['model_id']}",
        "",
        f"- Profile: {model['total_params_b']}B total, {model.get('active_params_b')}B active, {model['layers']} layers, hidden {model['hidden_size']}.",
        f"- Teacher bf16/fp16 weights: about {mem['teacher_weights_fp16_or_bf16']} GB before runtime overhead.",
        f"- Teacher inference headroom floor: about {mem['minimum_teacher_inference_headroom']} GB.",
        f"- Teacher+student training floor: about {mem['teacher_plus_student_training_floor']} GB before fragmentation, optimizer policy, and activations.",
        f"- Requested sequence length: {requested['sequence_length']} tokens.",
        "",
        "## Recommended path",
        guidance["recommended_path"],
        "",
        "## Local",
        guidance["local"]["summary"],
        "",
        "## Remote GPU",
        remote["summary"],
        "",
        "## API / Worker",
        api["summary"],
        "",
        "## Next steps",
    ]
    lines.extend(f"- {step}" for step in guidance["next_steps"])
    if model["notes"]:
        lines.extend(["", "## Model notes"])
        lines.extend(f"- {note}" for note in model["notes"])
    return "\n".join(lines)


def _profile_from_config(cfg: SlimderConfig, architecture: dict[str, Any] | None = None) -> ModelProfile:
    known = _known_profile(cfg.teacher.model_id_or_path)
    if known is not None:
        return known
    arch = architecture or {}
    total = float(arch.get("total_params", 0) or 0) / 1_000_000_000
    if total <= 0:
        total = _fallback_total_params_b(cfg)
    moe_layers = arch.get("moe_layers") or []
    first_moe = moe_layers[0] if moe_layers else {}
    return ModelProfile(
        model_id=cfg.teacher.model_id_or_path,
        family=str(arch.get("model_type") or "unknown"),
        total_params_b=round(total, 3),
        active_params_b=None,
        hidden_size=int(arch.get("hidden_size") or cfg.compression.target.hidden_size or 0),
        layers=int(arch.get("num_layers") or 0),
        routed_experts=first_moe.get("num_routed_experts"),
        top_k=first_moe.get("top_k"),
        shared_experts=first_moe.get("num_shared_experts"),
        native_context=None,
        notes=("No built-in profile matched; estimates use loaded architecture or conservative config fallback.",),
    )


def _known_profile(model_id: str) -> ModelProfile | None:
    key = model_id.replace("\\", "/").rstrip("/").lower()
    return KNOWN_PROFILES.get(key) or KNOWN_PROFILES.get(key.split("/")[-1])


def _fallback_total_params_b(cfg: SlimderConfig) -> float:
    if cfg.teacher.load_mode == "tiny":
        return 0.001
    return max(1.0, float(cfg.compression.target.hidden_size) * max(1, cfg.compression.target.routed_experts) / 1_000)


def _weight_memory(total_params_b: float) -> dict[str, float]:
    params = total_params_b * 1_000_000_000
    return {
        "fp16": params * 2 / 1_000_000_000,
        "fp32": params * 4 / 1_000_000_000,
        "int8": params / 1_000_000_000,
        "int4": params * 0.5 / 1_000_000_000,
    }


def _estimate_target_total_params_b(profile: ModelProfile, cfg: SlimderConfig) -> float:
    target = cfg.compression.target
    hidden_ratio = (target.hidden_size / profile.hidden_size) if profile.hidden_size else 1.0
    layer_ratio = max(1, profile.layers - target.remove_last_n_layers) / profile.layers if profile.layers else 1.0
    expert_ratio = (target.routed_experts / profile.routed_experts) if profile.routed_experts else 1.0
    sparse_weight_ratio = 0.65 * expert_ratio + 0.35 * hidden_ratio
    return max(0.001, profile.total_params_b * layer_ratio * min(1.0, sparse_weight_ratio))


def _local_fit_notes(profile: ModelProfile, cfg: SlimderConfig, weight_memory: dict[str, float]) -> dict[str, Any]:
    if cfg.teacher.load_mode == "tiny":
        return {"status": "ok", "summary": "Tiny mode is intended for CPU smoke tests and local development."}
    if profile.total_params_b >= 30:
        return {
            "status": "not_recommended_for_full_framework",
            "summary": (
                "Local full-framework compression/distillation is not recommended for this model unless the machine has "
                "multiple high-memory GPUs. Config-only recommend/analyze is fine locally; quantized inference may fit on "
                "workstations, but it does not provide the full logits needed for paper-faithful KD."
            ),
            "minimum_vram_for_teacher_bf16_gb": round(weight_memory["fp16"] * 1.25, 1),
            "quantized_inference_note": "A 4-bit runtime can be useful for manual inspection, but Slimder should use remote GPUs for compression/distillation.",
        }
    return {
        "status": "possible",
        "summary": "Local runs may be practical if bf16/fp16 teacher weights plus activation headroom fit in available VRAM.",
        "minimum_vram_for_teacher_bf16_gb": round(weight_memory["fp16"] * 1.25, 1),
    }


def _remote_guidance(profile: ModelProfile, cfg: SlimderConfig, teacher_mem: dict[str, float], student_mem: dict[str, float]) -> dict[str, Any]:
    if profile.total_params_b >= 30:
        min_gpu = "2x80GB GPUs for smoke-scale full-logit work; 4x80GB is a more realistic starting point."
        recommended = "4-8x80GB GPUs for meaningful calibration/distillation; 8xH100/A100-class for long runs."
    else:
        min_gpu = "1x24-48GB GPU may work for smoke-scale runs depending on sequence length."
        recommended = "1-2 high-memory GPUs for iterative development."
    return {
        "status": "recommended",
        "summary": (
            f"Use SSH, SkyPilot, or the Worker API for this model. Minimum: {min_gpu} "
            f"Recommended: {recommended}"
        ),
        "minimum": min_gpu,
        "recommended": recommended,
        "teacher_weight_gb": round(teacher_mem["fp16"], 1),
        "student_weight_gb": round(student_mem["fp16"], 1),
    }


def _api_guidance(cfg: SlimderConfig) -> dict[str, Any]:
    if cfg.project.paper_faithful:
        summary = (
            "Paper-faithful distillation requires exact full-vocabulary teacher logits. A generic chat-completions API is not enough. "
            "Use online local logits on remote GPUs, an exact full-logit cache, or a Worker API endpoint that returns full logits."
        )
        status = "full_logits_required"
    else:
        summary = (
            "Augmented mode may use non-paper-faithful services for inspection or auxiliary workflows, but manifests must label that path explicitly."
        )
        status = "augmented_allowed"
    if cfg.kd.teacher_mode == "remote_worker_full_logits":
        summary += " Current config is already set to the remote full-logit worker mode."
    return {"status": status, "summary": summary}


def _recommended_path(cfg: SlimderConfig, profile: ModelProfile) -> str:
    if cfg.teacher.load_mode == "tiny":
        return "Run locally on CPU for smoke validation."
    if profile.total_params_b >= 30:
        return (
            "Start locally with config-only recommend/dry-run, then launch through SkyPilot/SSH/Worker on high-memory GPUs. "
            "Use self-hosted full-logit teacher access for KD; reserve chat APIs for non-paper-faithful auxiliary workflows."
        )
    return "Use local GPU if the dry-run/preflight confirms memory headroom; otherwise use SSH or SkyPilot."


def _next_steps(cfg: SlimderConfig) -> list[str]:
    steps = [
        "Run `slimder analyze --config <cfg> --config-only --json` to verify architecture without loading weights.",
        "Run `slimder recommend --config <cfg> --config-only --write-config <cfg> --json` to materialize a compression plan.",
        "Run `slimder run <cfg> --dry-run --json` and inspect compute_guidance before launching.",
    ]
    if cfg.teacher.load_mode == "transformers" and cfg.teacher.model_id_or_path != "dummy-hf-moe":
        steps.append("Prefer `slimder launch --backend skypilot` or `--backend ssh` for the real run; avoid local full-model execution unless explicitly provisioned.")
    if cfg.project.paper_faithful:
        steps.append("Keep `kd.teacher_mode` on `online_full_logits`, `offline_full_logits_cache`, or `remote_worker_full_logits` with exact full distributions.")
    return steps

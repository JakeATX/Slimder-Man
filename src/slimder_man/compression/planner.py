from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StagePlan:
    stage: int
    remove_last_n_layers: int
    hidden_size: int
    routed_experts: int | None
    top_k: int | None
    tokens: int


@dataclass(frozen=True)
class ParameterEstimate:
    total_params: int
    active_params: int
    memory_bytes: int
    memory_gib: float


def validate_target(
    architecture: dict,
    hidden_size: int,
    remove_last_n_layers: int,
    routed_experts: int,
    routed_top_k: int,
    hidden_multiple: int = 128,
) -> None:
    source_hidden = int(architecture["hidden_size"])
    source_layers = int(architecture["num_layers"])
    source_experts = _source_routed_experts(architecture)
    source_top_k = _source_top_k(architecture)

    if hidden_size <= 0 or hidden_size > source_hidden:
        raise ValueError("hidden_size must be positive and no larger than the source hidden size")
    if hidden_size % hidden_multiple != 0:
        raise ValueError("hidden_size must satisfy the hidden multiple constraint")
    if remove_last_n_layers < 0 or remove_last_n_layers >= source_layers:
        raise ValueError("remove_last_n_layers must leave at least one layer")
    if routed_experts <= 0 or routed_experts > source_experts:
        raise ValueError("routed_experts must be positive and no larger than the source routed experts")
    if routed_top_k <= 0 or routed_top_k > source_top_k:
        raise ValueError("routed_top_k must be positive and no larger than the source top_k")
    if routed_top_k > routed_experts:
        raise ValueError("routed_top_k must not exceed routed_experts")


def estimate_parameters(
    architecture: dict,
    hidden_size: int,
    remove_last_n_layers: int,
    routed_experts: int,
    routed_top_k: int,
    bytes_per_param: int = 2,
) -> ParameterEstimate:
    source_hidden = int(architecture["hidden_size"])
    source_layers = int(architecture["num_layers"])
    target_layers = source_layers - remove_last_n_layers
    source_experts = _source_routed_experts(architecture)
    source_top_k = _source_top_k(architecture)
    shared_experts = _source_shared_experts(architecture)
    vocab_size = int(architecture["vocab_size"])
    tied_embeddings = bool(architecture.get("tied_embeddings", False))

    source_total_formula = _param_formula(
        vocab_size,
        source_hidden,
        source_layers,
        source_experts,
        source_experts,
        shared_experts,
        tied_embeddings,
    )
    target_total_formula = _param_formula(
        vocab_size,
        hidden_size,
        target_layers,
        routed_experts,
        routed_experts,
        shared_experts,
        tied_embeddings,
    )
    source_active_formula = _param_formula(
        vocab_size,
        source_hidden,
        source_layers,
        source_experts,
        source_top_k,
        shared_experts,
        tied_embeddings,
    )
    target_active_formula = _param_formula(
        vocab_size,
        hidden_size,
        target_layers,
        routed_experts,
        routed_top_k,
        shared_experts,
        tied_embeddings,
    )

    total_params = _scaled_estimate(int(architecture.get("total_params") or 0), source_total_formula, target_total_formula)
    active_anchor = int(architecture.get("active_params_estimate") or 0)
    active_params = _scaled_estimate(active_anchor, source_active_formula, target_active_formula) if active_anchor else target_active_formula
    active_params = min(active_params, total_params)
    memory_bytes = total_params * bytes_per_param
    return ParameterEstimate(total_params, active_params, memory_bytes, round(memory_bytes / 1024**3, 3))


def progressive_plan(schedule: str, stages: int, total_tokens: int, token_split: list[float], teacher_layers: int, remove_last_n: int, start_hidden: int, target_hidden: int, hidden_multiple: int = 128, target_experts: int | None = None, target_top_k: int | None = None) -> list[StagePlan]:
    if stages == 1 or schedule == "one_stage":
        return [StagePlan(1, remove_last_n, target_hidden, target_experts, target_top_k, int(total_tokens * token_split[0]))]
    half_depth = remove_last_n // 2
    half_hidden = start_hidden - (((start_hidden - target_hidden) // 2) // hidden_multiple) * hidden_multiple
    if schedule == "depth_first":
        stage1 = StagePlan(1, half_depth, start_hidden, None, None, int(total_tokens * token_split[0]))
    elif schedule == "width_first":
        stage1 = StagePlan(1, 0, half_hidden, None, None, int(total_tokens * token_split[0]))
    elif schedule == "joint":
        stage1 = StagePlan(1, half_depth, half_hidden, None, None, int(total_tokens * token_split[0]))
    else:
        raise ValueError(f"Unsupported progressive schedule {schedule}")
    stage2 = StagePlan(2, remove_last_n, target_hidden, target_experts, target_top_k, total_tokens - stage1.tokens)
    return [stage1, stage2]


def _source_routed_experts(architecture: dict) -> int:
    moe_layers = architecture.get("moe_layers") or []
    if moe_layers:
        return max(int(layer["num_routed_experts"]) for layer in moe_layers)
    return int(architecture.get("routed_experts") or architecture.get("num_routed_experts") or 1)


def _source_shared_experts(architecture: dict) -> int:
    moe_layers = architecture.get("moe_layers") or []
    if moe_layers:
        return max(int(layer.get("num_shared_experts", 0)) for layer in moe_layers)
    return int(architecture.get("shared_experts") or architecture.get("num_shared_experts") or 0)


def _source_top_k(architecture: dict) -> int:
    moe_layers = architecture.get("moe_layers") or []
    if moe_layers:
        return max(int(layer["top_k"]) for layer in moe_layers)
    return int(architecture.get("routed_top_k") or architecture.get("top_k") or 1)


def _param_formula(
    vocab_size: int,
    hidden_size: int,
    layers: int,
    routed_experts: int,
    active_routed_experts: int,
    shared_experts: int,
    tied_embeddings: bool,
) -> int:
    embedding_multiplier = 1 if tied_embeddings else 2
    embedding_params = vocab_size * hidden_size * embedding_multiplier
    dense_layer_params = hidden_size * hidden_size * 2
    expert_params = (active_routed_experts + shared_experts) * hidden_size * hidden_size
    router_params = hidden_size * routed_experts
    return embedding_params + layers * (dense_layer_params + expert_params + router_params)


def _scaled_estimate(anchor: int, source_formula: int, target_formula: int) -> int:
    if anchor <= 0 or source_formula <= 0:
        return target_formula
    return max(1, round(anchor * (target_formula / source_formula)))

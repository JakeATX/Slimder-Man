from __future__ import annotations

from dataclasses import asdict
from typing import Any

from slimder_man.adapters.base import ArchitectureInfo, MoELayerInfo


def describe_config_architecture(config: Any) -> dict:
    hidden = _first_int(config, ("hidden_size", "n_embd", "d_model"))
    vocab = _first_int(config, ("vocab_size",))
    layers = _first_int(config, ("num_hidden_layers", "n_layer", "num_layers"))
    experts = _first_int(config, ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"), default=1)
    shared = _shared_expert_count(config)
    top_k = _first_int(config, ("num_experts_per_tok", "moe_top_k", "top_k", "num_experts_per_token"), default=1)
    block_kinds = _block_kinds(config, layers)
    mtp_depths = _first_int(config, ("mtp_depth", "mtp_depths", "num_nextn_predict_layers"), default=0)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    info = ArchitectureInfo(
        model_type=str(getattr(config, "model_type", config.__class__.__name__)),
        total_params=0,
        active_params_estimate=None,
        hidden_size=hidden,
        vocab_size=vocab,
        num_layers=layers,
        block_kinds=block_kinds,
        num_full_attention_layers=sum(1 for kind in block_kinds if kind == "full_attention"),
        num_linear_attention_layers=sum(1 for kind in block_kinds if kind == "linear_attention"),
        moe_layers=[
            MoELayerInfo(layer_idx=idx, num_routed_experts=experts, num_shared_experts=shared, top_k=top_k)
            for idx in range(layers)
        ],
        has_mtp=mtp_depths > 0,
        mtp_depths=mtp_depths,
        tied_embeddings=tied,
        dtype_summary={},
        tensor_name_map={},
    )
    data = asdict(info)
    data["moe_layers"] = [asdict(layer) for layer in info.moe_layers]
    data["source"] = "config_only"
    data["analysis_scope"] = "config_only_structural"
    data["calibration_status"] = "not_run"
    data["weights_loaded"] = False
    data["parameter_count_status"] = "not_available_without_weights"
    data["mtp_detection_status"] = "detected_from_config_fields_only"
    return data


def _first_int(config: Any, names: tuple[str, ...], default: int = 0) -> int:
    for name in names:
        value = getattr(config, name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return default


def _shared_expert_count(config: Any) -> int:
    explicit = _first_int(config, ("num_shared_experts", "n_shared_experts"), default=-1)
    if explicit >= 0:
        return explicit
    intermediate = _first_int(config, ("shared_expert_intermediate_size", "shared_experts_intermediate_size"), default=0)
    return 1 if intermediate > 0 else 0


def _block_kinds(config: Any, layers: int) -> list[str]:
    raw = getattr(config, "layer_types", None) or getattr(config, "layers_block_type", None)
    if isinstance(raw, (list, tuple)) and raw:
        values = [_normalize_block_kind(str(item)) for item in raw]
        return [values[idx % len(values)] for idx in range(layers)]
    pattern = getattr(config, "block_pattern", None)
    if isinstance(pattern, (list, tuple)) and pattern:
        values = [_normalize_block_kind(str(item)) for item in pattern]
        return [values[idx % len(values)] for idx in range(layers)]
    return ["unknown" for _ in range(layers)]


def _normalize_block_kind(value: str) -> str:
    text = value.lower()
    if "linear" in text or "delta" in text:
        return "linear_attention"
    if "full" in text or "attention" in text:
        return "full_attention"
    return text or "unknown"

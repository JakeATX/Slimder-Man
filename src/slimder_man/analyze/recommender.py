from __future__ import annotations

from dataclasses import dataclass, asdict

from slimder_man.config.presets import PRESETS
from slimder_man.compression.planner import estimate_parameters, validate_target


PRESET_POLICIES = {
    "conservative_20": {"hidden_ratio": 0.90, "depth_ratio": 0.10, "expert_ratio": 0.80, "top_k_ratio": 1.00},
    "balanced_50": {"hidden_ratio": 0.75, "depth_ratio": 0.20, "expert_ratio": 0.50, "top_k_ratio": 0.80},
    "aggressive_80": {"hidden_ratio": 0.625, "depth_ratio": 0.33, "expert_ratio": 0.25, "top_k_ratio": 0.60},
    "extreme_90": {"hidden_ratio": 0.50, "depth_ratio": 0.50, "expert_ratio": 0.125, "top_k_ratio": 0.40},
}


@dataclass
class Candidate:
    preset: str
    candidate_id: str
    hidden_size: int
    remove_last_n_layers: int
    routed_experts: int
    routed_top_k: int
    schedule: str
    estimated_total_params: int
    estimated_active_params: int
    estimated_memory_bytes: int
    estimated_memory_gib: float
    risk: str
    transforms: dict


def recommend(architecture: dict, preset: str = "balanced_50", max_candidates: int = 8) -> list[dict]:
    if preset == "all":
        candidates: list[dict] = []
        for name in PRESETS:
            candidates.extend(recommend(architecture, name, max_candidates=max_candidates))
        return candidates
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset}")
    base = PRESETS[preset]
    source_hidden = int(architecture["hidden_size"])
    source_layers = int(architecture["num_layers"])
    source_experts = _source_routed_experts(architecture)
    source_top_k = _source_top_k(architecture)
    hidden_multiple = _hidden_multiple(architecture)
    target = _target_for_preset(preset, architecture, hidden_multiple)
    risks = {
        "conservative_20": "low",
        "balanced_50": "medium",
        "slimqwen_anchor": "paper-supported anchor",
        "aggressive_80": "high",
        "extreme_90": "very high",
    }
    candidates = []
    seen: set[tuple[int, int, int, int]] = set()
    raw_candidates = []
    for hidden in _hidden_options(source_hidden, target["hidden_size"], hidden_multiple):
        for remove_last in _depth_options(source_layers, target["remove_last_n_layers"]):
            for experts in _expert_options(source_experts, target["routed_experts"]):
                for top_k in _top_k_options(source_top_k, target["routed_top_k"]):
                    key = (hidden, remove_last, experts, top_k)
                    if key in seen:
                        continue
                    try:
                        validate_target(architecture, hidden, remove_last, experts, top_k, hidden_multiple)
                    except ValueError:
                        continue
                    seen.add(key)
                    raw_candidates.append(key)
    raw_candidates.sort(key=lambda item: _candidate_sort_key(item, target))
    for idx, (hidden, remove_last, experts, top_k) in enumerate(_select_diverse_candidates(raw_candidates, target, max_candidates), start=1):
        estimate = estimate_parameters(architecture, hidden, remove_last, experts, top_k)
        candidates.append(
            Candidate(
                preset=preset,
                candidate_id=f"{preset}_{idx}",
                hidden_size=hidden,
                remove_last_n_layers=remove_last,
                routed_experts=experts,
                routed_top_k=top_k,
                schedule=str(base["schedule"]),
                estimated_total_params=estimate.total_params,
                estimated_active_params=estimate.active_params,
                estimated_memory_bytes=estimate.memory_bytes,
                estimated_memory_gib=estimate.memory_gib,
                risk=risks[preset],
                transforms=_transforms(architecture, hidden, remove_last, experts, top_k, hidden_multiple),
            )
        )
    return [asdict(c) for c in candidates]


def _source_routed_experts(architecture: dict) -> int:
    moe_layers = architecture.get("moe_layers") or []
    if moe_layers:
        return max(int(layer["num_routed_experts"]) for layer in moe_layers)
    return int(architecture.get("routed_experts") or architecture.get("num_routed_experts") or 1)


def _source_top_k(architecture: dict) -> int:
    moe_layers = architecture.get("moe_layers") or []
    if moe_layers:
        return max(int(layer["top_k"]) for layer in moe_layers)
    return int(architecture.get("routed_top_k") or architecture.get("top_k") or 1)


def _hidden_multiple(architecture: dict) -> int:
    configured = int(architecture.get("hidden_size_multiple") or 0)
    if configured > 0:
        return configured
    return 128 if int(architecture["hidden_size"]) >= 128 else 4


def _target_for_preset(preset: str, architecture: dict, hidden_multiple: int) -> dict:
    source_hidden = int(architecture["hidden_size"])
    source_layers = int(architecture["num_layers"])
    source_experts = _source_routed_experts(architecture)
    source_top_k = _source_top_k(architecture)
    base = PRESETS[preset]

    if preset == "slimqwen_anchor":
        target = {
            "hidden_size": 1536 if source_hidden >= 2048 else min(source_hidden, int(base["hidden_size"])),
            "remove_last_n_layers": 12 if source_layers >= 48 else min(source_layers - 1, int(base["remove_last_n_layers"])),
            "routed_experts": 256 if source_experts >= 512 else min(source_experts, int(base["routed_experts"])),
            "routed_top_k": 8 if source_top_k >= 10 else min(source_top_k, int(base["routed_top_k"])),
        }
    elif source_hidden >= 128:
        policy = PRESET_POLICIES[preset]
        target = {
            "hidden_size": _floor_to_multiple(round(source_hidden * float(policy["hidden_ratio"])), hidden_multiple),
            "remove_last_n_layers": min(source_layers - 1, max(0, round(source_layers * float(policy["depth_ratio"])))),
            "routed_experts": max(1, round(source_experts * float(policy["expert_ratio"]))),
            "routed_top_k": max(1, round(source_top_k * float(policy["top_k_ratio"]))),
        }
    else:
        target = {
            "hidden_size": min(source_hidden, int(base["hidden_size"])),
            "remove_last_n_layers": min(source_layers - 1, int(base["remove_last_n_layers"])),
            "routed_experts": min(source_experts, int(base["routed_experts"])),
            "routed_top_k": min(source_top_k, int(base["routed_top_k"])),
        }

    target["hidden_size"] = _floor_to_multiple(min(target["hidden_size"], source_hidden), hidden_multiple)
    target["remove_last_n_layers"] = min(max(0, target["remove_last_n_layers"]), source_layers - 1)
    target["routed_experts"] = min(max(1, target["routed_experts"]), source_experts)
    target["routed_top_k"] = min(max(1, target["routed_top_k"]), source_top_k, target["routed_experts"])
    return target


def _floor_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def _hidden_options(source_hidden: int, target_hidden: int, multiple: int) -> list[int]:
    midpoint = _floor_to_multiple((source_hidden + target_hidden) // 2, multiple)
    values = [target_hidden, target_hidden + multiple, target_hidden - multiple, midpoint, source_hidden]
    return _unique_sorted_valid(values, lambda value: value > 0 and value <= source_hidden and value % multiple == 0)


def _depth_options(source_layers: int, target_remove_last: int) -> list[int]:
    values = [target_remove_last, max(0, target_remove_last - max(1, target_remove_last // 3)), min(source_layers - 1, target_remove_last + max(1, target_remove_last // 3)), 0]
    return _unique_sorted_valid(values, lambda value: 0 <= value < source_layers)


def _expert_options(source_experts: int, target_experts: int) -> list[int]:
    step = max(1, (source_experts - target_experts) // 2)
    values = [target_experts, min(source_experts, target_experts + step), max(1, target_experts - step), source_experts]
    return _unique_sorted_valid(values, lambda value: 0 < value <= source_experts)


def _top_k_options(source_top_k: int, target_top_k: int) -> list[int]:
    values = [target_top_k, min(source_top_k, target_top_k + 1), max(1, target_top_k - 1), source_top_k]
    return _unique_sorted_valid(values, lambda value: 0 < value <= source_top_k)


def _unique_sorted_valid(values: list[int], valid) -> list[int]:
    result = []
    seen = set()
    for value in values:
        if value in seen or not valid(value):
            continue
        seen.add(value)
        result.append(value)
    return result


def _candidate_sort_key(item: tuple[int, int, int, int], target: dict) -> tuple[int, int, int, int, int]:
    hidden, remove_last, experts, top_k = item
    distance = (
        abs(hidden - target["hidden_size"])
        + abs(remove_last - target["remove_last_n_layers"]) * 10_000
        + abs(experts - target["routed_experts"]) * 100
        + abs(top_k - target["routed_top_k"]) * 1_000
    )
    return (distance, -hidden, remove_last, -experts, -top_k)


def _select_diverse_candidates(candidates: list[tuple[int, int, int, int]], target: dict, limit: int) -> list[tuple[int, int, int, int]]:
    target_tuple = (
        target["hidden_size"],
        target["remove_last_n_layers"],
        target["routed_experts"],
        target["routed_top_k"],
    )
    selected = []
    if target_tuple in candidates:
        selected.append(target_tuple)
    for dim in range(4):
        variant = next((item for item in candidates if item[dim] != target_tuple[dim] and item not in selected), None)
        if variant is not None:
            selected.append(variant)
    for item in candidates:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)
    return selected[:limit]


def _transforms(architecture: dict, hidden: int, remove_last: int, experts: int, top_k: int, hidden_multiple: int) -> dict:
    source_layers = int(architecture["num_layers"])
    return {
        "width": {"from_hidden_size": int(architecture["hidden_size"]), "to_hidden_size": hidden, "hidden_multiple": hidden_multiple},
        "depth": {"from_layers": source_layers, "remove_last_n_layers": remove_last, "to_layers": source_layers - remove_last},
        "experts": {"from_routed_experts": _source_routed_experts(architecture), "to_routed_experts": experts},
        "router": {"from_top_k": _source_top_k(architecture), "to_top_k": top_k},
    }

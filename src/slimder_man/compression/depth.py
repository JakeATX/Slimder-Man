from __future__ import annotations

from math import floor


def compute_depth_keep_indices(num_layers: int, remove_last_n: int) -> list[int]:
    if remove_last_n < 0:
        raise ValueError("remove_last_n must be non-negative")
    if remove_last_n >= num_layers:
        raise ValueError("Cannot drop all transformer layers")
    return list(range(num_layers - remove_last_n))


def resolve_remove_last_n(num_layers: int, remove_last_n: int, depth_remove_fraction: float | None = None) -> int:
    if depth_remove_fraction is None:
        return remove_last_n
    if depth_remove_fraction < 0 or depth_remove_fraction >= 1:
        raise ValueError("depth_remove_fraction must be >= 0 and < 1")
    return floor(num_layers * depth_remove_fraction)

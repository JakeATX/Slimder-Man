from __future__ import annotations


def compute_depth_keep_indices(num_layers: int, remove_last_n: int) -> list[int]:
    if remove_last_n < 0:
        raise ValueError("remove_last_n must be non-negative")
    if remove_last_n >= num_layers:
        raise ValueError("Cannot drop all transformer layers")
    return list(range(num_layers - remove_last_n))

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QuantItem:
    name: str
    size: int
    saliency: float
    protected_bits: int | None = None


def allocate_bits(items: list[QuantItem], allowed_bits: list[int], target_avg_bits: float) -> dict[str, int]:
    if not items:
        return {}
    min_bits = min(allowed_bits)
    max_bits = max(allowed_bits)
    total_size = sum(i.size for i in items)
    budget = total_size * target_avg_bits
    protected_cost = sum(i.size * i.protected_bits for i in items if i.protected_bits is not None)
    free = [i for i in items if i.protected_bits is None]
    min_free_cost = sum(i.size * min_bits for i in free)
    if protected_cost + min_free_cost > budget + 1e-9:
        min_feasible_avg = (protected_cost + min_free_cost) / max(1, total_size)
        protected = [
            f"{i.name}(size={i.size}, protected_bits={i.protected_bits})"
            for i in items
            if i.protected_bits is not None
        ]
        protected_text = ", ".join(protected) if protected else "none"
        raise ValueError(
            "Infeasible quantization bit budget: "
            f"target_avg_bits={target_avg_bits:.3f}, minimum_feasible_avg_bits={min_feasible_avg:.3f}, "
            f"min_unprotected_bits={min_bits}, protected_modules={protected_text}"
        )
    result = {i.name: (i.protected_bits if i.protected_bits is not None else min_bits) for i in items}
    remaining = budget - protected_cost - min_free_cost
    for item in sorted(free, key=lambda x: x.saliency, reverse=True):
        for bits in sorted([b for b in allowed_bits if b > result[item.name]], reverse=True):
            extra = item.size * (bits - result[item.name])
            if extra <= remaining + 1e-9:
                result[item.name] = bits
                remaining -= extra
                break
    # Monotonic repair for equal sizes: higher saliency cannot get fewer bits.
    ordered = sorted(free, key=lambda x: x.saliency, reverse=True)
    for a, b in zip(ordered, ordered[1:]):
        if a.size == b.size and result[a.name] < result[b.name]:
            result[a.name] = result[b.name]
    return result

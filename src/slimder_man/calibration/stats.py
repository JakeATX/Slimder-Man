from __future__ import annotations

import torch


def expert_frequency(topk_indices: torch.Tensor, num_experts: int) -> torch.Tensor:
    counts = torch.zeros(num_experts, dtype=torch.float64)
    flat = topk_indices.reshape(-1)
    if flat.numel():
        counts.scatter_add_(0, flat.cpu(), torch.ones_like(flat, dtype=torch.float64, device="cpu"))
        counts /= topk_indices.shape[0]
    return counts


def expert_soft_importance(topk_indices: torch.Tensor, topk_weights: torch.Tensor, num_experts: int) -> torch.Tensor:
    scores = torch.zeros(num_experts, dtype=torch.float64)
    for slot in range(topk_indices.shape[1]):
        idx = topk_indices[:, slot].cpu()
        weights = topk_weights[:, slot].detach().cpu().to(torch.float64)
        scores.scatter_add_(0, idx, weights)
    if topk_indices.shape[0] > 0:
        scores /= topk_indices.shape[0]
    return scores


def expert_reap_importance(topk_indices: torch.Tensor, topk_weights: torch.Tensor, output_norm2: torch.Tensor, num_experts: int) -> torch.Tensor:
    """REAP-style importance as mean gate-weighted output norm over assigned tokens.

    Unselected experts remain exactly zero. This intentionally uses the
    assigned-token divisor, not total-token normalization.
    """
    scores = torch.zeros(num_experts, dtype=torch.float64)
    counts = torch.zeros(num_experts, dtype=torch.float64)
    for slot in range(topk_indices.shape[1]):
        idx = topk_indices[:, slot].cpu()
        weights = topk_weights[:, slot].detach().cpu().to(torch.float64)
        norm = output_norm2.detach().cpu().to(torch.float64).gather(1, idx.unsqueeze(1)).squeeze(1)
        scores.scatter_add_(0, idx, weights * norm)
        counts.scatter_add_(0, idx, torch.ones_like(weights, dtype=torch.float64))
    mask = counts > 0
    scores[mask] /= counts[mask]
    return scores


def expert_reap_numerator_counts(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    output_norm2: torch.Tensor,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = torch.zeros(num_experts, dtype=torch.float64)
    counts = torch.zeros(num_experts, dtype=torch.float64)
    for slot in range(topk_indices.shape[1]):
        idx = topk_indices[:, slot].cpu()
        weights = topk_weights[:, slot].detach().cpu().to(torch.float64)
        norm = output_norm2.detach().cpu().to(torch.float64).gather(1, idx.unsqueeze(1)).squeeze(1)
        numerator.scatter_add_(0, idx, weights * norm)
        counts.scatter_add_(0, idx, torch.ones_like(weights, dtype=torch.float64))
    return numerator, counts


def finalize_reap_importance(numerator: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    scores = numerator.clone()
    mask = counts > 0
    scores[mask] /= counts[mask]
    scores[~mask] = 0
    return scores


def cosine_similarity_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    x = matrix.to(torch.float64)
    num = x.T @ x
    norms = torch.sqrt(torch.clamp(torch.diag(num), min=0))
    denom = norms[:, None] * norms[None, :]
    sim = torch.zeros_like(num)
    mask = denom > 0
    sim[mask] = num[mask] / denom[mask]
    nonzero = norms > 0
    sim[nonzero, nonzero] = 1.0
    return sim.to(torch.float32)


def streaming_cosine(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        return torch.empty(0, 0)
    n = chunks[0].shape[1]
    num = torch.zeros(n, n, dtype=torch.float64)
    norms = torch.zeros(n, dtype=torch.float64)
    for chunk in chunks:
        x = chunk.to(torch.float64)
        num += x.T @ x
        norms += (x**2).sum(dim=0)
    denom = torch.sqrt(norms[:, None] * norms[None, :])
    sim = torch.zeros_like(num)
    mask = denom > 0
    sim[mask] = num[mask] / denom[mask]
    nonzero = norms > 0
    sim[nonzero, nonzero] = 1.0
    return sim.to(torch.float32)

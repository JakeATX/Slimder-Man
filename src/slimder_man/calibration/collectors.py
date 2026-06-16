from __future__ import annotations

from dataclasses import dataclass

import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.calibration.stats import (
    expert_frequency,
    expert_reap_importance,
    expert_soft_importance,
    streaming_cosine,
)


@dataclass
class CalibrationResult:
    hidden_scores: torch.Tensor
    per_layer_hidden_scores: list[torch.Tensor]
    expert_frequency: list[torch.Tensor]
    expert_soft: list[torch.Tensor]
    expert_reap: list[torch.Tensor]
    expert_similarity: list[torch.Tensor]
    representation: str = "post_softmax_topk_weights"


def collect_tiny_calibration(model: TinyMoEForCausalLM, batches: list[torch.Tensor]) -> CalibrationResult:
    hidden_size = model.config.hidden_size
    hidden_sums = [torch.zeros(hidden_size, dtype=torch.float64) for _ in range(len(model.layers) * 2 + 1)]
    hidden_counts = [0 for _ in hidden_sums]
    expert_freq = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
    expert_soft = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
    expert_reap = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
    weight_chunks = [[] for _ in model.layers]

    handles = []
    norm_modules = []
    for block in model.layers:
        norm_modules.extend([block.input_layernorm, block.post_attention_layernorm])
    norm_modules.append(model.norm)

    def make_hook(i: int):
        def hook(_module, _inputs, output):
            y = output.detach().cpu().to(torch.float64)
            hidden_sums[i].add_(y.abs().sum(dim=(0, 1)))
            hidden_counts[i] += y.shape[0] * y.shape[1]

        return hook

    for i, norm in enumerate(norm_modules):
        handles.append(norm.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        for batch in batches:
            _ = model(batch)
            for layer_idx, block in enumerate(model.layers):
                moe = block.moe
                assert moe.last_topk_indices is not None and moe.last_topk_weights is not None and moe.last_expert_output_norm2 is not None
                topi = moe.last_topk_indices.reshape(-1, moe.last_topk_indices.shape[-1])
                topw = moe.last_topk_weights.reshape(-1, moe.last_topk_weights.shape[-1])
                norm2 = moe.last_expert_output_norm2.reshape(-1, moe.num_routed_experts)
                expert_freq[layer_idx] += expert_frequency(topi, moe.num_routed_experts)
                expert_soft[layer_idx] += expert_soft_importance(topi, topw, moe.num_routed_experts)
                expert_reap[layer_idx] += expert_reap_importance(topi, topw, norm2, moe.num_routed_experts)
                dense_weights = torch.zeros(topi.shape[0], moe.num_routed_experts)
                for slot in range(topi.shape[1]):
                    dense_weights.scatter_add_(1, topi[:, slot : slot + 1].cpu(), topw[:, slot : slot + 1].cpu())
                weight_chunks[layer_idx].append(dense_weights)
    for h in handles:
        h.remove()
    denom = max(1, len(batches))
    per_hidden = [s / max(1, c) for s, c in zip(hidden_sums, hidden_counts)]
    global_scores = sum(per_hidden)
    return CalibrationResult(
        hidden_scores=global_scores.to(torch.float32),
        per_layer_hidden_scores=[x.to(torch.float32) for x in per_hidden],
        expert_frequency=[(x / denom).to(torch.float32) for x in expert_freq],
        expert_soft=[(x / denom).to(torch.float32) for x in expert_soft],
        expert_reap=[(x / denom).to(torch.float32) for x in expert_reap],
        expert_similarity=[streaming_cosine(chunks) for chunks in weight_chunks],
    )


def hidden_keep_indices(scores: torch.Tensor, target_hidden_size: int) -> torch.Tensor:
    if target_hidden_size <= 0 or target_hidden_size > scores.numel():
        raise ValueError("target_hidden_size must be within current hidden size")
    return torch.topk(scores, k=target_hidden_size).indices.sort().values

from __future__ import annotations

from dataclasses import dataclass

import torch

from slimder_man.adapters.registry import get_adapter
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
    router_logits_similarity: list[torch.Tensor] | None = None
    router_weights_similarity: list[torch.Tensor] | None = None
    expert_outputs_similarity: list[torch.Tensor] | None = None
    representation: str = "post_softmax_topk_weights"


def collect_tiny_calibration(model: TinyMoEForCausalLM, batches: list[torch.Tensor]) -> CalibrationResult:
    hidden_size = model.config.hidden_size
    hidden_sums = [torch.zeros(hidden_size, dtype=torch.float64) for _ in range(len(model.layers) * 2 + 1)]
    hidden_counts = [0 for _ in hidden_sums]
    expert_freq = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
    expert_soft = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
    expert_reap = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
    weight_chunks = [[] for _ in model.layers]
    logit_chunks = [[] for _ in model.layers]

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
                logit_chunks[layer_idx].append(moe.last_router_logits.reshape(-1, moe.num_routed_experts).cpu())
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
        router_logits_similarity=[streaming_cosine(chunks) for chunks in logit_chunks],
        router_weights_similarity=[streaming_cosine(chunks) for chunks in weight_chunks],
        expert_outputs_similarity=[streaming_cosine(chunks) for chunks in weight_chunks],
    )


def hidden_keep_indices(scores: torch.Tensor, target_hidden_size: int) -> torch.Tensor:
    if target_hidden_size <= 0 or target_hidden_size > scores.numel():
        raise ValueError("target_hidden_size must be within current hidden size")
    return torch.topk(scores, k=target_hidden_size).indices.sort().values


def collect_calibration(model, batches: list[torch.Tensor], adapter=None) -> CalibrationResult:
    """Generic calibration collector for MoE fixtures exposing routing traces.

    The collector is intentionally structural and model-agnostic: adapters find
    RMSNorms and MoE modules; MoE modules are expected to expose routing traces
    (`last_router_logits`, `last_topk_indices`, `last_topk_weights`,
    `last_expert_output_norm2`) after forward. This matches the dummy HF fixture
    and many testable HF-style wrappers without depending on TinyMoEForCausalLM.
    """

    if isinstance(model, TinyMoEForCausalLM):
        return collect_tiny_calibration(model, batches)
    adapter = adapter or get_adapter(model)
    hidden_size = adapter.describe_architecture(model).hidden_size
    norms = adapter.iter_rmsnorms(model)
    moes = adapter.iter_moe_layers(model)
    hidden_sums = [torch.zeros(hidden_size, dtype=torch.float64) for _ in norms]
    hidden_counts = [0 for _ in norms]
    expert_freq = [torch.zeros(len(adapter.get_routed_experts(moe)), dtype=torch.float64) for moe in moes]
    expert_soft = [torch.zeros(len(adapter.get_routed_experts(moe)), dtype=torch.float64) for moe in moes]
    expert_reap = [torch.zeros(len(adapter.get_routed_experts(moe)), dtype=torch.float64) for moe in moes]
    weight_chunks = [[] for _ in moes]
    logit_chunks = [[] for _ in moes]
    handles = []

    def make_hook(i: int):
        def hook(_module, _inputs, output):
            y = output.detach().cpu().to(torch.float64)
            if y.shape[-1] == hidden_size:
                hidden_sums[i].add_(y.abs().sum(dim=tuple(range(y.ndim - 1))))
                hidden_counts[i] += int(torch.tensor(y.shape[:-1]).prod().item())

        return hook

    for i, norm in enumerate(norms):
        handles.append(norm.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        for batch in batches:
            _ = model(input_ids=batch)
            for layer_idx, moe in enumerate(moes):
                topi = getattr(moe, "last_topk_indices", None)
                topw = getattr(moe, "last_topk_weights", None)
                logits = getattr(moe, "last_router_logits", None)
                norm2 = getattr(moe, "last_expert_output_norm2", None)
                if topi is None or topw is None or logits is None or norm2 is None:
                    raise ValueError(f"MoE layer {layer_idx} did not expose routing traces during calibration")
                n = len(adapter.get_routed_experts(moe))
                topi = topi.reshape(-1, topi.shape[-1])
                topw = topw.reshape(-1, topw.shape[-1])
                logits = logits.reshape(-1, n)
                norm2 = norm2.reshape(-1, n)
                expert_freq[layer_idx] += expert_frequency(topi, n)
                expert_soft[layer_idx] += expert_soft_importance(topi, topw, n)
                expert_reap[layer_idx] += expert_reap_importance(topi, topw, norm2, n)
                dense_weights = torch.zeros(topi.shape[0], n)
                for slot in range(topi.shape[1]):
                    dense_weights.scatter_add_(1, topi[:, slot : slot + 1].cpu(), topw[:, slot : slot + 1].cpu())
                weight_chunks[layer_idx].append(dense_weights)
                logit_chunks[layer_idx].append(logits.cpu())
    for h in handles:
        h.remove()
    denom = max(1, len(batches))
    per_hidden = [s / max(1, c) for s, c in zip(hidden_sums, hidden_counts)]
    global_scores = sum(per_hidden) if per_hidden else torch.ones(hidden_size)
    return CalibrationResult(
        hidden_scores=global_scores.to(torch.float32),
        per_layer_hidden_scores=[x.to(torch.float32) for x in per_hidden],
        expert_frequency=[(x / denom).to(torch.float32) for x in expert_freq],
        expert_soft=[(x / denom).to(torch.float32) for x in expert_soft],
        expert_reap=[(x / denom).to(torch.float32) for x in expert_reap],
        expert_similarity=[streaming_cosine(chunks) for chunks in weight_chunks],
        router_logits_similarity=[streaming_cosine(chunks) for chunks in logit_chunks],
        router_weights_similarity=[streaming_cosine(chunks) for chunks in weight_chunks],
        expert_outputs_similarity=[streaming_cosine(chunks) for chunks in weight_chunks],
    )

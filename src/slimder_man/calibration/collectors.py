from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from slimder_man.adapters.registry import get_adapter
from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.calibration.stats import (
    expert_frequency,
    expert_reap_numerator_counts,
    expert_soft_importance,
    finalize_reap_importance,
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
    expert_layer_indices: list[int] | None = None
    representation: str = "post_softmax_topk_weights"


def collect_tiny_calibration(model: TinyMoEForCausalLM, batches: list[torch.Tensor]) -> CalibrationResult:
    hidden_size = model.config.hidden_size
    hidden_sums = [torch.zeros(hidden_size, dtype=torch.float64) for _ in range(len(model.layers) * 2 + 1)]
    hidden_counts = [0 for _ in hidden_sums]
    expert_freq = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
    expert_soft = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
    expert_reap_num = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
    expert_reap_count = [torch.zeros(model.config.num_routed_experts, dtype=torch.float64) for _ in model.layers]
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
                reap_num, reap_count = expert_reap_numerator_counts(topi, topw, norm2, moe.num_routed_experts)
                expert_reap_num[layer_idx] += reap_num
                expert_reap_count[layer_idx] += reap_count
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
        expert_reap=[finalize_reap_importance(num, count).to(torch.float32) for num, count in zip(expert_reap_num, expert_reap_count, strict=True)],
        expert_similarity=[streaming_cosine(chunks) for chunks in weight_chunks],
        router_logits_similarity=[streaming_cosine(chunks) for chunks in logit_chunks],
        router_weights_similarity=[streaming_cosine(chunks) for chunks in weight_chunks],
        expert_layer_indices=list(range(len(model.layers))),
    )


def hidden_keep_indices(scores: torch.Tensor, target_hidden_size: int) -> torch.Tensor:
    if target_hidden_size <= 0 or target_hidden_size > scores.numel():
        raise ValueError("target_hidden_size must be within current hidden size")
    return torch.topk(scores, k=target_hidden_size).indices.sort().values


def collect_calibration(model, batches: list[torch.Tensor], adapter=None) -> CalibrationResult:
    """Generic calibration collector for adapter-discovered MoE models.

    Adapters find RMSNorms and MoE modules. If modules expose routing traces
    after forward, those traces are used. Otherwise the collector falls back to
    hooks on the MoE input and router, then recomputes selected expert output
    norms structurally for REAP-style scores.
    """

    if isinstance(model, TinyMoEForCausalLM):
        return collect_tiny_calibration(model, batches)
    adapter = adapter or get_adapter(model)
    architecture = adapter.describe_architecture(model)
    expert_layer_indices = [int(info.layer_idx) for info in architecture.moe_layers]
    hidden_size = architecture.hidden_size
    norms = adapter.iter_rmsnorms(model)
    moes = adapter.iter_moe_layers(model)
    hidden_sums = [torch.zeros(hidden_size, dtype=torch.float64) for _ in norms]
    hidden_counts = [0 for _ in norms]
    expert_counts = [_adapter_expert_count(adapter, moe) for moe in moes]
    expert_freq = [torch.zeros(n, dtype=torch.float64) for n in expert_counts]
    expert_soft = [torch.zeros(n, dtype=torch.float64) for n in expert_counts]
    expert_reap_num = [torch.zeros(n, dtype=torch.float64) for n in expert_counts]
    expert_reap_count = [torch.zeros(n, dtype=torch.float64) for n in expert_counts]
    weight_grams = [torch.zeros(n, n, dtype=torch.float64) for n in expert_counts]
    weight_norms = [torch.zeros(n, dtype=torch.float64) for n in expert_counts]
    logit_grams = [torch.zeros(n, n, dtype=torch.float64) for n in expert_counts]
    logit_norms = [torch.zeros(n, dtype=torch.float64) for n in expert_counts]
    top_ks = [
        min(expert_counts[idx], max(1, int(architecture.moe_layers[idx].top_k)))
        if idx < len(architecture.moe_layers) and architecture.moe_layers[idx].top_k
        else _moe_top_k(moe, expert_counts[idx])
        for idx, moe in enumerate(moes)
    ]
    handles = []
    trace_states = [SimpleNamespace(hidden=None, router_logits=None) for _ in moes]
    used_hook_fallback = False

    def make_hook(i: int):
        def hook(_module, _inputs, output):
            y = output.detach().cpu().to(torch.float64)
            if y.shape[-1] == hidden_size:
                hidden_sums[i].add_(y.abs().sum(dim=tuple(range(y.ndim - 1))))
                hidden_counts[i] += int(torch.tensor(y.shape[:-1]).prod().item())

        return hook

    for i, norm in enumerate(norms):
        handles.append(norm.register_forward_hook(make_hook(i)))
    for i, moe in enumerate(moes):
        handles.append(moe.register_forward_pre_hook(_make_moe_input_hook(trace_states[i], hidden_size)))
        handles.append(adapter.get_router(moe).register_forward_hook(_make_router_hook(trace_states[i])))
    was_training = bool(getattr(model, "training", False))
    try:
        if hasattr(model, "eval"):
            model.eval()
        with torch.no_grad():
            for batch in batches:
                for state in trace_states:
                    state.hidden = None
                    state.router_logits = None
                _ = model(input_ids=batch)
                for layer_idx, moe in enumerate(moes):
                    n = expert_counts[layer_idx]
                    traces = _resolve_moe_traces(adapter, moe, trace_states[layer_idx], n, top_ks[layer_idx])
                    if traces.used_hook_fallback:
                        used_hook_fallback = True
                    topi = traces.topi.reshape(-1, traces.topi.shape[-1])
                    topw = traces.topw.reshape(-1, traces.topw.shape[-1])
                    logits = traces.logits.reshape(-1, n)
                    norm2 = traces.norm2.reshape(-1, n)
                    expert_freq[layer_idx] += expert_frequency(topi, n)
                    expert_soft[layer_idx] += expert_soft_importance(topi, topw, n)
                    reap_num, reap_count = expert_reap_numerator_counts(topi, topw, norm2, n)
                    expert_reap_num[layer_idx] += reap_num
                    expert_reap_count[layer_idx] += reap_count
                    dense_weights = torch.zeros(topi.shape[0], n)
                    for slot in range(topi.shape[1]):
                        dense_weights.scatter_add_(1, topi[:, slot : slot + 1].cpu(), topw[:, slot : slot + 1].cpu())
                    _update_cosine_accumulator(weight_grams[layer_idx], weight_norms[layer_idx], dense_weights)
                    _update_cosine_accumulator(logit_grams[layer_idx], logit_norms[layer_idx], logits)
    finally:
        for h in handles:
            h.remove()
        if was_training and hasattr(model, "train"):
            model.train()
    denom = max(1, len(batches))
    per_hidden = [s / max(1, c) for s, c in zip(hidden_sums, hidden_counts)]
    global_scores = sum(per_hidden) if per_hidden else torch.ones(hidden_size)
    return CalibrationResult(
        hidden_scores=global_scores.to(torch.float32),
        per_layer_hidden_scores=[x.to(torch.float32) for x in per_hidden],
        expert_frequency=[(x / denom).to(torch.float32) for x in expert_freq],
        expert_soft=[(x / denom).to(torch.float32) for x in expert_soft],
        expert_reap=[finalize_reap_importance(num, count).to(torch.float32) for num, count in zip(expert_reap_num, expert_reap_count, strict=True)],
        expert_similarity=[_cosine_from_accumulator(gram, norms) for gram, norms in zip(weight_grams, weight_norms, strict=True)],
        router_logits_similarity=[_cosine_from_accumulator(gram, norms) for gram, norms in zip(logit_grams, logit_norms, strict=True)],
        router_weights_similarity=[_cosine_from_accumulator(gram, norms) for gram, norms in zip(weight_grams, weight_norms, strict=True)],
        expert_layer_indices=expert_layer_indices,
        representation="router_hook_recomputed_expert_outputs" if used_hook_fallback else "post_softmax_topk_weights",
    )


def _make_moe_input_hook(state: SimpleNamespace, hidden_size: int):
    def hook(_module, inputs):
        value = _first_tensor(inputs)
        if value is None or value.shape[-1] != hidden_size:
            return
        state.hidden = value.detach().reshape(-1, hidden_size)

    return hook


def _make_router_hook(state: SimpleNamespace):
    def hook(_module, _inputs, output):
        value = _first_tensor(output)
        if value is not None:
            state.router_logits = value.detach()

    return hook


def _first_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _resolve_moe_traces(adapter, moe, state: SimpleNamespace, num_experts: int, top_k: int) -> SimpleNamespace:
    topi = getattr(moe, "last_topk_indices", None)
    topw = getattr(moe, "last_topk_weights", None)
    logits = getattr(moe, "last_router_logits", None)
    norm2 = getattr(moe, "last_expert_output_norm2", None)
    if topi is not None and topw is not None and logits is not None and norm2 is not None:
        return SimpleNamespace(topi=topi, topw=topw, logits=logits, norm2=norm2, used_hook_fallback=False)
    if state.hidden is None or state.router_logits is None:
        raise ValueError("MoE layer did not expose routing traces and hook fallback did not capture router inputs/logits")
    logits = state.router_logits.reshape(-1, num_experts)
    if state.hidden.shape[0] != logits.shape[0]:
        raise ValueError(
            f"MoE hook fallback captured mismatched token counts: hidden={state.hidden.shape[0]}, "
            f"router_logits={logits.shape[0]}"
        )
    topv, topi = torch.topk(logits, k=top_k, dim=-1)
    topw = torch.softmax(topv, dim=-1)
    norm2 = _selected_expert_output_norm2(adapter, moe, state.hidden, topi, num_experts)
    return SimpleNamespace(topi=topi, topw=topw, logits=logits, norm2=norm2, used_hook_fallback=True)


def _moe_top_k(moe, num_experts: int) -> int:
    for name in ("top_k", "num_experts_per_tok", "moe_top_k"):
        value = getattr(moe, name, None)
        if value is not None:
            return min(num_experts, max(1, int(value)))
    return min(num_experts, 2)


def _adapter_expert_count(adapter, moe) -> int:
    count_fn = getattr(adapter, "routed_expert_count", None)
    if callable(count_fn):
        return int(count_fn(moe))
    return len(adapter.get_routed_experts(moe))


def _update_cosine_accumulator(gram: torch.Tensor, norms: torch.Tensor, matrix: torch.Tensor) -> None:
    x = matrix.detach().cpu().to(torch.float64)
    gram.add_(x.T @ x)
    norms.add_((x**2).sum(dim=0))


def _cosine_from_accumulator(gram: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
    denom = torch.sqrt(norms[:, None] * norms[None, :])
    sim = torch.zeros_like(gram)
    mask = denom > 0
    sim[mask] = gram[mask] / denom[mask]
    nonzero = norms > 0
    sim[nonzero, nonzero] = 1.0
    return sim.to(torch.float32)


def _selected_expert_output_norm2(adapter, moe, hidden: torch.Tensor, topi: torch.Tensor, num_experts: int) -> torch.Tensor:
    direct = getattr(adapter, "selected_expert_output_norm2", None)
    if callable(direct):
        result = direct(moe, hidden, topi, num_experts)
        if result is not None:
            return result
    experts = adapter.get_routed_experts(moe)
    norm2 = torch.zeros(topi.shape[0], num_experts, dtype=torch.float32)
    for slot in range(topi.shape[1]):
        for expert_idx, expert in enumerate(experts):
            mask = topi[:, slot] == expert_idx
            if not mask.any():
                continue
            expert_input = hidden[mask]
            try:
                device = next(expert.parameters()).device
            except StopIteration:
                device = expert_input.device
            output = _first_tensor(expert(expert_input.to(device)))
            if output is None:
                continue
            norm2[mask.cpu(), expert_idx] = output.detach().float().pow(2).sum(dim=-1).cpu()
    return norm2

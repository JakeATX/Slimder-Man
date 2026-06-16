from __future__ import annotations

import torch
from torch import nn

from slimder_man.adapters.tiny import TinyExpert, TinyMoEForCausalLM


def _slice_linear(linear: nn.Linear, keep_in: torch.Tensor | None = None, keep_out: torch.Tensor | None = None) -> nn.Linear:
    weight = linear.weight.detach().clone()
    bias = linear.bias.detach().clone() if linear.bias is not None else None
    if keep_out is not None:
        weight = weight.index_select(0, keep_out)
        if bias is not None:
            bias = bias.index_select(0, keep_out)
    if keep_in is not None:
        weight = weight.index_select(1, keep_in)
    new = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None)
    with torch.no_grad():
        new.weight.copy_(weight)
        if bias is not None and new.bias is not None:
            new.bias.copy_(bias)
    return new


def _slice_norm_weight(norm: nn.Module, keep_idx: torch.Tensor) -> None:
    norm.weight = nn.Parameter(norm.weight.detach().clone().index_select(0, keep_idx))


def _slice_expert(expert: TinyExpert, keep_idx: torch.Tensor) -> TinyExpert:
    intermediate = expert.w1.out_features
    new = TinyExpert(len(keep_idx), intermediate)
    new.w1 = _slice_linear(expert.w1, keep_in=keep_idx)
    new.w3 = _slice_linear(expert.w3, keep_in=keep_idx)
    new.w2 = _slice_linear(expert.w2, keep_out=keep_idx)
    return new


def slice_tiny_hidden(model: TinyMoEForCausalLM, keep_idx: torch.Tensor) -> None:
    keep_idx = keep_idx.cpu().to(torch.long)
    was_tied = model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()
    old_weight = model.embed_tokens.weight.detach().clone()
    new_embed = nn.Embedding(model.config.vocab_size, len(keep_idx))
    with torch.no_grad():
        new_embed.weight.copy_(old_weight.index_select(1, keep_idx))
    model.embed_tokens = new_embed
    for block in model.layers:
        _slice_norm_weight(block.input_layernorm, keep_idx)
        block.attn = _slice_linear(block.attn, keep_in=keep_idx, keep_out=keep_idx)
        _slice_norm_weight(block.post_attention_layernorm, keep_idx)
        moe = block.moe
        moe.hidden_size = len(keep_idx)
        moe.router = _slice_linear(moe.router, keep_in=keep_idx)
        moe.experts = nn.ModuleList([_slice_expert(expert, keep_idx) for expert in moe.experts])
        moe.shared_experts = nn.ModuleList([_slice_expert(expert, keep_idx) for expert in moe.shared_experts])
        if moe.shared_gate is not None:
            moe.shared_gate = _slice_linear(moe.shared_gate, keep_in=keep_idx)
    _slice_norm_weight(model.norm, keep_idx)
    if was_tied:
        model.lm_head = nn.Linear(len(keep_idx), model.config.vocab_size, bias=False)
        model.lm_head.weight = model.embed_tokens.weight
    else:
        model.lm_head = _slice_linear(model.lm_head, keep_in=keep_idx)
    model.mtp_heads = nn.ModuleList([_slice_linear(head, keep_in=keep_idx) for head in model.mtp_heads])
    model.config.hidden_size = len(keep_idx)


def select_keep_indices(scores: torch.Tensor, target_hidden_size: int) -> torch.Tensor:
    if target_hidden_size <= 0 or target_hidden_size > scores.numel():
        raise ValueError("target_hidden_size must be within current hidden size")
    return torch.topk(scores, k=target_hidden_size).indices.sort().values

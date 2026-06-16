from __future__ import annotations

import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM


def validate_tiny_model(model: TinyMoEForCausalLM) -> list[str]:
    errors: list[str] = []
    for name, p in model.named_parameters():
        if any(dim == 0 for dim in p.shape):
            errors.append(f"{name} has zero dimension")
    input_ids = torch.randint(0, model.config.vocab_size, (1, min(8, model.config.vocab_size)))
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    if not torch.isfinite(out.logits).all():
        errors.append("logits are not finite")
    if out.loss is None or not torch.isfinite(out.loss):
        errors.append("loss is not finite")
    if out.logits.shape[-1] != model.config.vocab_size:
        errors.append("logits vocab shape does not match config")
    for block in model.layers:
        if block.moe.top_k > block.moe.num_routed_experts:
            errors.append("router top_k exceeds routed experts")
        if block.moe.router.out_features != len(block.moe.experts):
            errors.append("router rows do not match expert count")
    if model.config.tie_embeddings and model.lm_head.weight.data_ptr() != model.embed_tokens.weight.data_ptr():
        errors.append("tied embeddings are not tied")
    return errors

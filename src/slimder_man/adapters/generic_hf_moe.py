from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from .base import ArchitectureInfo, MoELayerInfo, count_parameters, dtype_summary


def _cfg(config: object | None, model: nn.Module) -> object | None:
    return config or getattr(model, "config", None)


def _first_int(obj: object | None, names: tuple[str, ...], default: int = 0) -> int:
    if obj is None:
        return default
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return int(value)
    return default


def _expert_container(moe: nn.Module) -> nn.ModuleList | nn.ModuleDict | None:
    experts = getattr(moe, "experts", None)
    if isinstance(experts, (nn.ModuleList, nn.ModuleDict)):
        return experts
    for attr in ("mlp", "moe"):
        nested = getattr(moe, attr, None)
        experts = getattr(nested, "experts", None)
        if isinstance(experts, (nn.ModuleList, nn.ModuleDict)):
            return experts
    return None


def _module_list(value: nn.ModuleList | nn.ModuleDict | nn.Module | None) -> list[nn.Module]:
    if isinstance(value, nn.ModuleDict):
        return list(value.values())
    if isinstance(value, nn.ModuleList):
        return list(value)
    if isinstance(value, nn.Module):
        return [value]
    return []


def _find_router(moe: nn.Module) -> nn.Linear | None:
    for attr in ("router", "gate", "gate_proj"):
        value = getattr(moe, attr, None)
        if isinstance(value, nn.Linear):
            return value
    for name, module in moe.named_children():
        if name.lower() in {"router", "gate", "gate_proj"} and isinstance(module, nn.Linear):
            return module
    return None


def is_structural_moe_layer(module: nn.Module) -> bool:
    experts = _expert_container(module)
    if experts is None or len(experts) == 0:
        return False
    router = _find_router(module)
    if router is None:
        return False
    return router.out_features >= len(experts)


def structural_moe_layers(model: nn.Module) -> list[nn.Module]:
    out: list[nn.Module] = []
    seen: set[int] = set()
    for _, module in model.named_modules():
        if id(module) in seen:
            continue
        if is_structural_moe_layer(module):
            out.append(module)
            seen.add(id(module))
    return out


class GenericHfMoeAdapter:
    """Structural adapter for HF-compatible MoE causal language models."""

    def match(self, model: nn.Module, config: object | None = None) -> bool:
        if structural_moe_layers(model):
            return True
        cfg = _cfg(config, model)
        model_type = str(getattr(cfg, "model_type", "")).lower()
        return "moe" in model_type and bool(self.iter_transformer_blocks(model))

    def describe_architecture(self, model: nn.Module, config: object | None = None) -> ArchitectureInfo:
        cfg = _cfg(config, model)
        layers = self.iter_transformer_blocks(model)
        moe_layers = self.iter_moe_layers(model)
        hidden = _first_int(cfg, ("hidden_size", "n_embd", "d_model"))
        vocab = _first_int(cfg, ("vocab_size",))
        block_kinds = [self.get_block_kind(block) for block in layers]
        emb = self._embedding_module(model)
        head = getattr(model, "lm_head", None)
        tied = bool(emb is not None and head is not None and getattr(emb, "weight", None) is getattr(head, "weight", None))
        return ArchitectureInfo(
            model_type=str(getattr(cfg, "model_type", model.__class__.__name__)),
            total_params=count_parameters(model),
            active_params_estimate=None,
            hidden_size=hidden,
            vocab_size=vocab,
            num_layers=len(layers),
            block_kinds=block_kinds,
            num_full_attention_layers=sum(1 for kind in block_kinds if kind == "full_attention"),
            num_linear_attention_layers=sum(1 for kind in block_kinds if kind == "linear_attention"),
            moe_layers=[
                MoELayerInfo(
                    layer_idx=self._layer_idx_for_moe(moe, layers, fallback=i),
                    num_routed_experts=len(self.get_routed_experts(moe)),
                    num_shared_experts=len(self.get_shared_experts(moe)),
                    top_k=self._top_k(moe, cfg),
                )
                for i, moe in enumerate(moe_layers)
            ],
            has_mtp=bool(self.get_mtp_modules(model)),
            mtp_depths=len(self.get_mtp_modules(model)),
            tied_embeddings=tied,
            dtype_summary=dtype_summary(model),
            tensor_name_map={name: name for name, _ in model.named_parameters()},
        )

    def iter_transformer_blocks(self, model: nn.Module) -> list[nn.Module]:
        for base in self._candidate_bases(model):
            for attr in ("layers", "h", "blocks"):
                value = getattr(base, attr, None)
                if isinstance(value, nn.ModuleList):
                    return list(value)
            decoder = getattr(base, "decoder", None)
            value = getattr(decoder, "layers", None)
            if isinstance(value, nn.ModuleList):
                return list(value)
        return []

    def get_block_kind(self, block: nn.Module) -> str:
        text = " ".join(name.lower() for name, _ in block.named_modules())
        if "linear_attention" in text or "deltanet" in text:
            return "linear_attention"
        if "attention" in text or "self_attn" in text or "attn" in text:
            return "full_attention"
        return "other"

    def iter_rmsnorms(self, model: nn.Module) -> list[nn.Module]:
        return [
            module
            for _, module in model.named_modules()
            if module.__class__.__name__.lower().endswith("rmsnorm")
            or (
                isinstance(module, nn.LayerNorm)
                and getattr(module, "normalized_shape", None)
                and len(module.normalized_shape) == 1
            )
        ]

    def iter_moe_layers(self, model: nn.Module) -> list[nn.Module]:
        return structural_moe_layers(model)

    def get_routed_experts(self, moe: nn.Module) -> list[nn.Module]:
        return _module_list(_expert_container(moe))

    def get_shared_experts(self, moe: nn.Module) -> list[nn.Module]:
        for attr in ("shared_experts", "shared_expert"):
            value = getattr(moe, attr, None)
            if value is not None:
                return _module_list(value)
        return []

    def get_router(self, moe: nn.Module) -> nn.Linear:
        router = _find_router(moe)
        if router is None:
            raise ValueError("Unable to locate router linear layer")
        return router

    def get_mtp_modules(self, model: nn.Module) -> list[nn.Module]:
        return [module for name, module in model.named_modules() if "mtp" in name.lower()]

    def slice_hidden_channels(self, model: nn.Module, keep_idx: torch.Tensor) -> None:
        raise NotImplementedError("Generic HF MoE structural adapter supports introspection only")

    def drop_blocks(self, model: nn.Module, keep_block_idx: list[int]) -> None:
        for base in self._candidate_bases(model):
            layers = getattr(base, "layers", None)
            if isinstance(layers, nn.ModuleList):
                base.layers = nn.ModuleList([layers[i] for i in keep_block_idx])
                return
            decoder = getattr(base, "decoder", None)
            layers = getattr(decoder, "layers", None)
            if isinstance(layers, nn.ModuleList):
                decoder.layers = nn.ModuleList([layers[i] for i in keep_block_idx])
                return
        raise ValueError("Unable to locate transformer block list")

    def replace_experts(self, moe: nn.Module, new_experts: list[nn.Module], router_rows: torch.Tensor, new_top_k: int) -> None:
        current = _expert_container(moe)
        if isinstance(current, nn.ModuleDict):
            moe.experts = nn.ModuleDict({str(i): expert for i, expert in enumerate(new_experts)})
        else:
            moe.experts = nn.ModuleList(new_experts)
        router = self.get_router(moe)
        new_router = nn.Linear(router_rows.shape[1], router_rows.shape[0], bias=router.bias is not None)
        with torch.no_grad():
            new_router.weight.copy_(router_rows.to(new_router.weight.dtype))
            if new_router.bias is not None:
                new_router.bias.zero_()
        if hasattr(moe, "router"):
            moe.router = new_router
        elif hasattr(moe, "gate"):
            moe.gate = new_router
        elif hasattr(moe, "gate_proj"):
            moe.gate_proj = new_router
        for attr in ("top_k", "num_experts_per_tok", "moe_top_k"):
            if hasattr(moe, attr):
                setattr(moe, attr, min(new_top_k, len(new_experts)))

    def update_config_after_compression(self, model: nn.Module, manifest: dict) -> None:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return
        cfg.num_hidden_layers = len(manifest["depth"]["kept_block_indices"])
        cfg.num_experts = manifest["target"]["routed_experts"]
        cfg.num_experts_per_tok = manifest["target"]["top_k"]
        if manifest["width"]["hidden_size_after"] != getattr(cfg, "hidden_size", manifest["width"]["hidden_size_after"]):
            cfg.hidden_size = manifest["width"]["hidden_size_after"]

    def save_pretrained(self, model: nn.Module, output_dir: str, manifest: dict | None = None) -> None:  # pragma: no cover
        if not hasattr(model, "save_pretrained"):
            raise NotImplementedError("Model does not expose save_pretrained")
        model.save_pretrained(output_dir)

    def _candidate_bases(self, model: nn.Module) -> Iterable[nn.Module]:
        yield model
        for attr in ("model", "transformer", "gpt_neox", "base_model"):
            value = getattr(model, attr, None)
            if isinstance(value, nn.Module):
                yield value

    def _embedding_module(self, model: nn.Module) -> nn.Module | None:
        for base in self._candidate_bases(model):
            for attr in ("embed_tokens", "wte", "word_embeddings"):
                value = getattr(base, attr, None)
                if isinstance(value, nn.Module):
                    return value
        return None

    def _top_k(self, moe: nn.Module, cfg: object | None) -> int:
        return _first_int(moe, ("top_k", "num_experts_per_tok", "moe_top_k")) or _first_int(
            cfg, ("num_experts_per_tok", "moe_top_k", "top_k")
        )

    def _layer_idx_for_moe(self, moe: nn.Module, layers: list[nn.Module], fallback: int) -> int:
        for idx, layer in enumerate(layers):
            if any(child is moe for child in layer.modules()):
                return idx
        return fallback

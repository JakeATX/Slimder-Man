from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ExpertContainerHandle:
    owner: nn.Module
    attr: str
    container: nn.ModuleList | nn.ModuleDict


@dataclass(frozen=True)
class RouterHandle:
    owner: nn.Module
    attr: str
    router: nn.Linear


def _expert_container_handle(moe: nn.Module) -> ExpertContainerHandle | None:
    experts = getattr(moe, "experts", None)
    if isinstance(experts, (nn.ModuleList, nn.ModuleDict)):
        return ExpertContainerHandle(moe, "experts", experts)
    for attr in ("mlp", "moe"):
        nested = getattr(moe, attr, None)
        experts = getattr(nested, "experts", None)
        if isinstance(nested, nn.Module) and isinstance(experts, (nn.ModuleList, nn.ModuleDict)):
            return ExpertContainerHandle(nested, "experts", experts)
    return None


def _expert_container(moe: nn.Module) -> nn.ModuleList | nn.ModuleDict | None:
    handle = _expert_container_handle(moe)
    return handle.container if handle is not None else None


def _module_list(value: nn.ModuleList | nn.ModuleDict | nn.Module | None) -> list[nn.Module]:
    if isinstance(value, nn.ModuleDict):
        return list(value.values())
    if isinstance(value, nn.ModuleList):
        return list(value)
    if isinstance(value, nn.Module):
        return [value]
    return []


def _router_handle(moe: nn.Module) -> RouterHandle | None:
    for attr in ("router", "gate", "gate_proj"):
        value = getattr(moe, attr, None)
        if isinstance(value, nn.Linear):
            return RouterHandle(moe, attr, value)
    for name, module in moe.named_children():
        if name.lower() in {"router", "gate", "gate_proj"} and isinstance(module, nn.Linear):
            return RouterHandle(moe, name, module)
    for container_attr in ("mlp", "moe"):
        nested = getattr(moe, container_attr, None)
        if not isinstance(nested, nn.Module):
            continue
        for attr in ("router", "gate", "gate_proj"):
            value = getattr(nested, attr, None)
            if isinstance(value, nn.Linear):
                return RouterHandle(nested, attr, value)
        for name, module in nested.named_children():
            if name.lower() in {"router", "gate", "gate_proj"} and isinstance(module, nn.Linear):
                return RouterHandle(nested, name, module)
    return None


def _find_router(moe: nn.Module) -> nn.Linear | None:
    handle = _router_handle(moe)
    return handle.router if handle is not None else None


def _slice_linear(module: nn.Linear, keep_idx: torch.Tensor, *, slice_in: bool, slice_out: bool) -> None:
    weight = module.weight.detach()
    bias_param = module.bias
    bias = bias_param.detach() if bias_param is not None else None
    index = keep_idx.to(weight.device)
    if slice_out:
        weight = weight.index_select(0, index)
        bias = bias.index_select(0, index.to(bias.device)) if bias is not None else None
        module.out_features = keep_idx.numel()
    if slice_in:
        weight = weight.index_select(1, index)
        module.in_features = keep_idx.numel()
    module.weight = nn.Parameter(weight.clone(), requires_grad=module.weight.requires_grad)
    if bias is not None and bias_param is not None:
        module.bias = nn.Parameter(bias.clone(), requires_grad=bias_param.requires_grad)


def _slice_norm(module: nn.Module, keep_idx: torch.Tensor) -> None:
    device_idx = keep_idx.to(module.weight.device)
    module.weight = nn.Parameter(module.weight.detach().index_select(0, device_idx).clone(), requires_grad=module.weight.requires_grad)
    if getattr(module, "bias", None) is not None:
        module.bias = nn.Parameter(
            module.bias.detach().index_select(0, keep_idx.to(module.bias.device)).clone(),
            requires_grad=module.bias.requires_grad,
        )
    if isinstance(module, nn.LayerNorm):
        module.normalized_shape = (keep_idx.numel(),)


def _linear_slice_rule(name: str, module: nn.Linear, hidden: int, *, is_router: bool = False) -> tuple[bool, bool] | None:
    leaf = name.rsplit(".", 1)[-1].lower()
    lowered = name.lower()
    is_attention = any(part in lowered for part in ("attn", "attention", "self_attn"))
    is_expert_context = any(part in lowered for part in (".experts.", ".shared_experts.", ".expert.", ".shared_expert."))

    if leaf == "lm_head":
        return module.in_features == hidden, False
    if leaf in {"q_proj", "k_proj", "v_proj", "query", "key", "value"} and is_attention:
        return module.in_features == hidden, False
    if leaf in {"o_proj", "out_proj"} and is_attention:
        return False, module.out_features == hidden
    if leaf in {"in_proj_qkvz", "in_proj_ba"}:
        return module.in_features == hidden, False
    if leaf == "shared_expert_gate":
        return module.in_features == hidden, False
    if leaf in {"up_proj", "w1", "w3"} and is_expert_context and module.out_features != hidden:
        return module.in_features == hidden, False
    if leaf == "gate_proj" and is_expert_context and module.out_features != hidden:
        return module.in_features == hidden, False
    if leaf in {"down_proj", "w2"} and is_expert_context:
        return False, module.out_features == hidden
    if leaf in {"router", "gate", "gate_proj"} and is_router and module.out_features != hidden:
        return module.in_features == hidden, False
    return None


def slice_structural_hidden_channels(model: nn.Module, keep_idx: torch.Tensor, hidden: int | None = None) -> None:
    """Slice hidden channels using named structural roles instead of all Linear shapes."""
    keep_idx = keep_idx.detach().cpu().to(torch.long)
    cfg = getattr(model, "config", None)
    old_hidden = int(hidden or getattr(cfg, "hidden_size", keep_idx.numel()))
    named = list(model.named_modules())
    embedding_heads: list[tuple[nn.Embedding, nn.Linear]] = []
    preserves_attention_width = False
    router_ids = {id(router) for moe in structural_moe_layers(model) if (router := _find_router(moe)) is not None}
    for _, embedding in named:
        if not isinstance(embedding, nn.Embedding) or embedding.weight.shape[1] != old_hidden:
            continue
        for _, head in named:
            if isinstance(head, nn.Linear) and head.weight is embedding.weight:
                embedding_heads.append((embedding, head))

    unsupported: list[str] = []
    for name, module in named:
        if isinstance(module, nn.Embedding) and module.weight.shape[1] == old_hidden:
            module.weight = nn.Parameter(
                module.weight.detach().index_select(1, keep_idx.to(module.weight.device)).clone(),
                requires_grad=module.weight.requires_grad,
            )
            module.embedding_dim = keep_idx.numel()
            continue
        if isinstance(module, nn.Linear):
            rule = _linear_slice_rule(name, module, old_hidden, is_router=id(module) in router_ids)
            if rule is None:
                if module.in_features == old_hidden or module.out_features == old_hidden:
                    unsupported.append(name or module.__class__.__name__)
                continue
            slice_in, slice_out = rule
            leaf = name.rsplit(".", 1)[-1].lower()
            if leaf in {"q_proj", "k_proj", "v_proj", "query", "key", "value"} and module.out_features == old_hidden:
                preserves_attention_width = True
            if leaf in {"o_proj", "out_proj"} and module.in_features == old_hidden:
                preserves_attention_width = True
            if slice_in or slice_out:
                _slice_linear(module, keep_idx, slice_in=slice_in, slice_out=slice_out)
            continue
        if (
            hasattr(module, "weight")
            and getattr(module.weight, "ndim", 0) == 1
            and module.weight.shape[0] == old_hidden
        ):
            _slice_norm(module, keep_idx)

    if unsupported:
        names = ", ".join(unsupported[:8])
        suffix = "" if len(unsupported) <= 8 else f", ... ({len(unsupported)} total)"
        raise ValueError(f"Unsupported hidden-size Linear modules for structural slicing: {names}{suffix}")

    for embedding, head in embedding_heads:
        head.weight = embedding.weight
        head.in_features = keep_idx.numel()

    if cfg is not None:
        cfg.hidden_size = keep_idx.numel()
        if preserves_attention_width and hasattr(cfg, "attention_hidden_size"):
            cfg.attention_hidden_size = old_hidden


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
    return [
        module
        for module in out
        if not any(other is not module and any(child is other for child in module.modules()) for other in out)
    ]


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
        slice_structural_hidden_channels(model, keep_idx)

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
        handle = _expert_container_handle(moe)
        current = handle.container if handle is not None else None
        if isinstance(current, nn.ModuleDict):
            replacement = nn.ModuleDict({str(i): expert for i, expert in enumerate(new_experts)})
        else:
            replacement = nn.ModuleList(new_experts)
        if handle is not None:
            setattr(handle.owner, handle.attr, replacement)
        else:
            moe.experts = replacement
        router = self.get_router(moe)
        new_router = nn.Linear(router_rows.shape[1], router_rows.shape[0], bias=router.bias is not None)
        with torch.no_grad():
            new_router.weight.copy_(router_rows.to(new_router.weight.dtype))
            if new_router.bias is not None:
                new_router.bias.zero_()
        handle = _router_handle(moe)
        if handle is not None:
            setattr(handle.owner, handle.attr, new_router)
        elif hasattr(moe, "router"):
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

    def save_pretrained(self, model: nn.Module, output_dir: str, manifest: dict | None = None) -> None:
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

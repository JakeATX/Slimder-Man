from __future__ import annotations

import torch
from torch import nn

from .base import ArchitectureInfo, MoELayerInfo, count_parameters, dtype_summary
from .generic_hf_moe import slice_structural_hidden_channels, structural_moe_layers


class Qwen3NextAdapter:
    """Best-effort Qwen3-Next introspection adapter.

    Full tensor surgery is adapter-scaffolded for v1; tiny-model tests exercise
    exact compression behavior. This adapter deliberately combines config
    fields, module names, and shapes instead of relying on one class name.
    """

    def match(self, model: nn.Module, config: object | None = None) -> bool:
        model_type = str(getattr(config, "model_type", "") or getattr(getattr(model, "config", None), "model_type", "")).lower()
        names = " ".join(name.lower() for name, _ in list(model.named_modules())[:500])
        return "qwen3" in model_type and ("next" in model_type or "gate" in names or "expert" in names)

    def describe_architecture(self, model: nn.Module, config: object | None = None) -> ArchitectureInfo:
        cfg = config or getattr(model, "config", None)
        hidden = int(getattr(cfg, "hidden_size", 0) or 0)
        vocab = int(getattr(cfg, "vocab_size", 0) or 0)
        layers = self.iter_transformer_blocks(model)
        moe_layers = self.iter_moe_layers(model)
        top_k = int(getattr(cfg, "num_experts_per_tok", getattr(cfg, "moe_top_k", 0)) or 0)
        n_experts = int(getattr(cfg, "num_experts", getattr(cfg, "n_routed_experts", 0)) or 0)
        shared = int(getattr(cfg, "num_shared_experts", 0) or 0)
        block_kinds = [self.get_block_kind(layer) for layer in layers]
        emb = getattr(model, "embed_tokens", None) or getattr(getattr(model, "model", None), "embed_tokens", None)
        head = getattr(model, "lm_head", None)
        tied = bool(emb is not None and head is not None and getattr(emb, "weight", None) is getattr(head, "weight", None))
        return ArchitectureInfo(
            model_type=str(getattr(cfg, "model_type", "qwen3_next")),
            total_params=count_parameters(model),
            active_params_estimate=None,
            hidden_size=hidden,
            vocab_size=vocab,
            num_layers=len(layers),
            block_kinds=block_kinds,
            num_full_attention_layers=sum(1 for k in block_kinds if k == "full_attention"),
            num_linear_attention_layers=sum(1 for k in block_kinds if k == "linear_attention"),
            moe_layers=[
                MoELayerInfo(
                    i,
                    n_experts or len(self.get_routed_experts(moe)),
                    shared or len(self.get_shared_experts(moe)),
                    top_k or int(getattr(moe, "top_k", getattr(moe, "num_experts_per_tok", getattr(moe, "moe_top_k", 0))) or 0),
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
        base = getattr(model, "model", model)
        for attr in ("layers", "h", "blocks"):
            value = getattr(base, attr, None)
            if isinstance(value, nn.ModuleList):
                return list(value)
        return []

    def get_block_kind(self, block: nn.Module) -> str:
        text = " ".join(name.lower() for name, _ in block.named_modules())
        if "deltanet" in text or "linear" in text and "attention" not in text:
            return "linear_attention"
        if "attention" in text or "self_attn" in text:
            return "full_attention"
        return "other"

    def iter_rmsnorms(self, model: nn.Module) -> list[nn.Module]:
        return [m for _, m in model.named_modules() if m.__class__.__name__.lower().endswith("rmsnorm")]

    def iter_moe_layers(self, model: nn.Module) -> list[nn.Module]:
        return structural_moe_layers(model)

    def get_routed_experts(self, moe: nn.Module) -> list[nn.Module]:
        experts = getattr(moe, "experts", [])
        if isinstance(experts, (dict, nn.ModuleDict)):
            return list(experts.values())
        return list(experts)

    def get_shared_experts(self, moe: nn.Module) -> list[nn.Module]:
        for attr in ("shared_experts", "shared_expert"):
            value = getattr(moe, attr, None)
            if value is None:
                continue
            if isinstance(value, nn.ModuleList):
                return list(value)
            if isinstance(value, nn.Module):
                return [value]
        return []

    def get_router(self, moe: nn.Module) -> nn.Linear:
        for attr in ("router", "gate", "gate_proj"):
            value = getattr(moe, attr, None)
            if isinstance(value, nn.Linear):
                return value
        raise ValueError("Unable to locate router linear layer")

    def get_mtp_modules(self, model: nn.Module) -> list[nn.Module]:
        return [m for name, m in model.named_modules() if "mtp" in name.lower()]

    def slice_hidden_channels(self, model: nn.Module, keep_idx):
        slice_structural_hidden_channels(model, keep_idx)

    def drop_blocks(self, model: nn.Module, keep_block_idx: list[int]) -> None:  # pragma: no cover
        base = getattr(model, "model", model)
        layers = getattr(base, "layers")
        setattr(base, "layers", nn.ModuleList([layers[i] for i in keep_block_idx]))

    def replace_experts(self, moe: nn.Module, new_experts: list[nn.Module], router_rows, new_top_k: int) -> None:  # pragma: no cover
        experts = getattr(moe, "experts", None)
        if isinstance(experts, nn.ModuleList):
            moe.experts = nn.ModuleList(new_experts)
        elif isinstance(experts, dict):
            moe.experts = nn.ModuleDict({str(i): expert for i, expert in enumerate(new_experts)})
        else:
            raise ValueError("Unsupported Qwen3-Next expert container")
        router = self.get_router(moe)
        new_router = nn.Linear(router_rows.shape[1], router_rows.shape[0], bias=router.bias is not None)
        with torch.no_grad():
            new_router.weight.copy_(router_rows.to(new_router.weight.dtype))
            if router.bias is not None and new_router.bias is not None:
                new_router.bias.zero_()
        if hasattr(moe, "router"):
            moe.router = new_router
        elif hasattr(moe, "gate"):
            moe.gate = new_router
        for attr in ("top_k", "num_experts_per_tok", "moe_top_k"):
            if hasattr(moe, attr):
                setattr(moe, attr, min(new_top_k, len(new_experts)))

    def update_config_after_compression(self, model: nn.Module, manifest: dict) -> None:  # pragma: no cover
        cfg = getattr(model, "config", None)
        if cfg is not None:
            cfg.hidden_size = manifest["target"]["hidden_size"]
            cfg.num_hidden_layers = len(manifest["depth"]["kept_block_indices"])
            cfg.num_experts = manifest["target"]["routed_experts"]
            cfg.num_experts_per_tok = manifest["target"]["top_k"]

    def save_pretrained(self, model: nn.Module, output_dir: str, manifest: dict | None = None) -> None:  # pragma: no cover
        model.save_pretrained(output_dir, safe_serialization=True)

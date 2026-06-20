from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import ArchitectureInfo, MoELayerInfo, count_parameters, dtype_summary
from .generic_hf_moe import slice_structural_hidden_channels, structural_moe_layers


class PackedQwenExpertSlice(nn.Module):
    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, act_fn: nn.Module | None = None):
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up_proj.detach().clone())
        self.down_proj = nn.Parameter(down_proj.detach().clone())
        self.act_fn = act_fn if act_fn is not None else nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = F.linear(x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        return F.linear(self.act_fn(gate) * up, self.down_proj)


def _is_packed_experts(module: nn.Module | None) -> bool:
    if module is None:
        return False
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    return isinstance(gate_up, torch.nn.Parameter) and isinstance(down, torch.nn.Parameter) and gate_up.ndim == 3 and down.ndim == 3


def _packed_experts(moe: nn.Module) -> nn.Module | None:
    experts = getattr(moe, "experts", None)
    return experts if _is_packed_experts(experts) else None


def _router_weight_module(moe: nn.Module) -> nn.Module | None:
    for attr in ("router", "gate", "gate_proj"):
        value = getattr(moe, attr, None)
        weight = getattr(value, "weight", None)
        if isinstance(value, nn.Module) and isinstance(weight, torch.nn.Parameter) and weight.ndim == 2:
            return value
    return None


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
        if layers and n_experts > 1 and not moe_layers:
            raise ValueError(
                "Qwen3-Next MoE block expected by config but no routed expert tensors matched; "
                "expected a ModuleList/ModuleDict experts container or packed gate_up_proj/down_proj tensors."
            )
        shared = self._shared_expert_count(cfg)
        block_kinds = self._block_kinds(cfg, layers)
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
                    self._layer_idx_for_moe(moe, layers, fallback=i),
                    n_experts or len(self.get_routed_experts(moe)),
                    max(shared, len(self.get_shared_experts(moe))),
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
        layers = structural_moe_layers(model)
        seen = {id(layer) for layer in layers}
        for _, module in model.named_modules():
            if id(module) in seen:
                continue
            packed = _packed_experts(module)
            router = _router_weight_module(module)
            if packed is None or router is None:
                continue
            if router.weight.shape[0] != packed.gate_up_proj.shape[0]:
                continue
            layers.append(module)
            seen.add(id(module))
        return layers

    def get_routed_experts(self, moe: nn.Module) -> list[nn.Module]:
        packed = _packed_experts(moe)
        if packed is not None:
            return [
                PackedQwenExpertSlice(packed.gate_up_proj[idx], packed.down_proj[idx], getattr(packed, "act_fn", None))
                for idx in range(packed.gate_up_proj.shape[0])
            ]
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

    def get_router(self, moe: nn.Module) -> nn.Module:
        router = _router_weight_module(moe)
        if router is not None:
            return router
        for attr in ("router", "gate", "gate_proj"):
            value = getattr(moe, attr, None)
            if isinstance(value, nn.Linear):
                return value
        raise ValueError("Unable to locate router weight module")

    def get_mtp_modules(self, model: nn.Module) -> list[nn.Module]:
        return [m for name, m in model.named_modules() if "mtp" in name.lower()]

    def slice_hidden_channels(self, model: nn.Module, keep_idx):
        keep = keep_idx.detach().cpu().to(torch.long)
        cfg = getattr(model, "config", None)
        old_hidden = int(getattr(cfg, "hidden_size", keep.numel()) or keep.numel())
        for moe in self.iter_moe_layers(model):
            router = _router_weight_module(moe)
            if router is not None and router.weight.shape[1] == old_hidden:
                router.weight = nn.Parameter(
                    router.weight.detach().index_select(1, keep.to(router.weight.device)).clone(),
                    requires_grad=router.weight.requires_grad,
                )
                for attr in ("hidden_dim", "hidden_size"):
                    if getattr(router, attr, None) == old_hidden:
                        setattr(router, attr, keep.numel())
            packed = _packed_experts(moe)
            if packed is not None:
                if packed.gate_up_proj.shape[2] == old_hidden:
                    packed.gate_up_proj = nn.Parameter(
                        packed.gate_up_proj.detach().index_select(2, keep.to(packed.gate_up_proj.device)).clone(),
                        requires_grad=packed.gate_up_proj.requires_grad,
                    )
                if packed.down_proj.shape[1] == old_hidden:
                    packed.down_proj = nn.Parameter(
                        packed.down_proj.detach().index_select(1, keep.to(packed.down_proj.device)).clone(),
                        requires_grad=packed.down_proj.requires_grad,
                    )
        slice_structural_hidden_channels(model, keep, hidden=old_hidden)
        for module in model.modules():
            if getattr(module, "hidden_size", None) == old_hidden:
                module.hidden_size = keep.numel()

    def drop_blocks(self, model: nn.Module, keep_block_idx: list[int]) -> None:
        base = getattr(model, "model", model)
        layers = getattr(base, "layers")
        setattr(base, "layers", nn.ModuleList([layers[i] for i in keep_block_idx]))
        cfg = getattr(model, "config", None)
        layer_types = getattr(cfg, "layer_types", None) if cfg is not None else None
        if isinstance(layer_types, list) and len(layer_types) == len(layers):
            cfg.layer_types = [layer_types[i] for i in keep_block_idx]

    def replace_experts(self, moe: nn.Module, new_experts: list[nn.Module], router_rows, new_top_k: int) -> None:
        packed = _packed_experts(moe)
        if packed is not None:
            gate_up = torch.stack([expert.gate_up_proj.detach() for expert in new_experts], dim=0).to(
                device=packed.gate_up_proj.device,
                dtype=packed.gate_up_proj.dtype,
            )
            down = torch.stack([expert.down_proj.detach() for expert in new_experts], dim=0).to(
                device=packed.down_proj.device,
                dtype=packed.down_proj.dtype,
            )
            packed.gate_up_proj = nn.Parameter(gate_up, requires_grad=packed.gate_up_proj.requires_grad)
            packed.down_proj = nn.Parameter(down, requires_grad=packed.down_proj.requires_grad)
            router = self.get_router(moe)
            router.weight = nn.Parameter(
                router_rows.to(device=router.weight.device, dtype=router.weight.dtype).clone(),
                requires_grad=router.weight.requires_grad,
            )
            for module in (moe, router, packed):
                for attr in ("num_experts", "n_routed_experts"):
                    if hasattr(module, attr):
                        setattr(module, attr, len(new_experts))
                for attr in ("hidden_dim", "hidden_size"):
                    if hasattr(module, attr) and attr in {"hidden_dim", "hidden_size"}:
                        current = getattr(module, attr)
                        if isinstance(current, int) and router.weight.shape[1] != current:
                            setattr(module, attr, router.weight.shape[1])
                for attr in ("top_k", "num_experts_per_tok", "moe_top_k"):
                    if hasattr(module, attr):
                        setattr(module, attr, min(new_top_k, len(new_experts)))
            return
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

    def update_config_after_compression(self, model: nn.Module, manifest: dict) -> None:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            cfg.hidden_size = manifest["target"]["hidden_size"]
            cfg.num_hidden_layers = len(manifest["depth"]["kept_block_indices"])
            cfg.num_experts = manifest["target"]["routed_experts"]
            cfg.num_experts_per_tok = manifest["target"]["top_k"]
            layer_types = getattr(cfg, "layer_types", None)
            if isinstance(layer_types, list) and len(layer_types) != cfg.num_hidden_layers:
                kept = manifest["depth"]["kept_block_indices"]
                if len(layer_types) > max(kept, default=-1):
                    cfg.layer_types = [layer_types[i] for i in kept]
                else:
                    cfg.layer_types = layer_types[: cfg.num_hidden_layers]

    def save_pretrained(self, model: nn.Module, output_dir: str, manifest: dict | None = None) -> None:
        model.save_pretrained(output_dir, safe_serialization=True)

    def _layer_idx_for_moe(self, moe: nn.Module, layers: list[nn.Module], fallback: int) -> int:
        for idx, layer in enumerate(layers):
            if any(child is moe for child in layer.modules()):
                return idx
        return fallback

    def _shared_expert_count(self, cfg: object | None) -> int:
        if cfg is None:
            return 0
        for name in ("num_shared_experts", "n_shared_experts"):
            value = getattr(cfg, name, None)
            if value is not None:
                return int(value)
        return 1 if int(getattr(cfg, "shared_expert_intermediate_size", 0) or 0) > 0 else 0

    def _block_kinds(self, cfg: object | None, layers: list[nn.Module]) -> list[str]:
        raw = getattr(cfg, "layer_types", None) if cfg is not None else None
        if isinstance(raw, (list, tuple)) and raw:
            values = [self._normalize_block_kind(str(item)) for item in raw]
            return [values[idx % len(values)] for idx in range(len(layers))]
        return [self.get_block_kind(layer) for layer in layers]

    def _normalize_block_kind(self, value: str) -> str:
        text = value.lower()
        if "linear" in text or "delta" in text:
            return "linear_attention"
        if "full" in text or "attention" in text:
            return "full_attention"
        return text or "other"

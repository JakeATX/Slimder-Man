from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
from torch import nn


@dataclass
class MoELayerInfo:
    layer_idx: int
    num_routed_experts: int
    num_shared_experts: int
    top_k: int


@dataclass
class ArchitectureInfo:
    model_type: str
    total_params: int
    active_params_estimate: int | None
    hidden_size: int
    vocab_size: int
    num_layers: int
    block_kinds: list[str]
    num_full_attention_layers: int
    num_linear_attention_layers: int
    moe_layers: list[MoELayerInfo]
    has_mtp: bool
    mtp_depths: int
    tied_embeddings: bool
    dtype_summary: dict[str, int]
    tensor_name_map: dict[str, str] = field(default_factory=dict)


class MoEModelAdapter(Protocol):
    def match(self, model: nn.Module, config: object | None = None) -> bool: ...
    def describe_architecture(self, model: nn.Module, config: object | None = None) -> ArchitectureInfo: ...
    def iter_transformer_blocks(self, model: nn.Module) -> list[nn.Module]: ...
    def get_block_kind(self, block: nn.Module) -> str: ...
    def iter_rmsnorms(self, model: nn.Module) -> list[nn.Module]: ...
    def iter_moe_layers(self, model: nn.Module) -> list[nn.Module]: ...
    def get_routed_experts(self, moe: nn.Module) -> list[nn.Module]: ...
    def get_shared_experts(self, moe: nn.Module) -> list[nn.Module]: ...
    def get_router(self, moe: nn.Module) -> nn.Linear: ...
    def get_mtp_modules(self, model: nn.Module) -> list[nn.Module]: ...
    def slice_hidden_channels(self, model: nn.Module, keep_idx: torch.Tensor) -> None: ...
    def drop_blocks(self, model: nn.Module, keep_block_idx: list[int]) -> None: ...
    def replace_experts(self, moe: nn.Module, new_experts: list[nn.Module], router_rows: torch.Tensor, new_top_k: int) -> None: ...
    def update_config_after_compression(self, model: nn.Module, manifest: dict) -> None: ...
    def save_pretrained(self, model: nn.Module, output_dir: str, manifest: dict | None = None) -> None: ...


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def dtype_summary(model: nn.Module) -> dict[str, int]:
    summary: dict[str, int] = {}
    for p in model.parameters():
        key = str(p.dtype)
        summary[key] = summary.get(key, 0) + p.numel()
    return summary

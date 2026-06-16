from __future__ import annotations

from dataclasses import asdict

from torch import nn

from slimder_man.adapters.registry import get_adapter


def describe_model(model: nn.Module, config: object | None = None) -> dict:
    adapter = get_adapter(model, config)
    info = adapter.describe_architecture(model, config)
    data = asdict(info)
    data["moe_layers"] = [asdict(x) for x in info.moe_layers]
    return data

from __future__ import annotations

from torch import nn

from .qwen3_next import Qwen3NextAdapter
from .tiny import TinyAdapter


ADAPTERS = [TinyAdapter(), Qwen3NextAdapter()]


def get_adapter(model: nn.Module, config: object | None = None):
    for adapter in ADAPTERS:
        if adapter.match(model, config):
            return adapter
    raise ValueError(f"Unsupported architecture: {model.__class__.__name__}")

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any


class OptionalQuantBackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class QuantBackendSpec:
    name: str
    module_names: tuple[str, ...]
    install_hint: str
    mode: str

    def available_modules(self) -> dict[str, bool]:
        return {module: importlib.util.find_spec(module) is not None for module in self.module_names}

    def is_available(self) -> bool:
        return all(self.available_modules().values())

    def status(self) -> dict[str, Any]:
        modules = self.available_modules()
        return {
            "backend": self.name,
            "mode": self.mode,
            "available": all(modules.values()),
            "modules": modules,
            "install_hint": self.install_hint,
        }

    def require_available(self) -> None:
        if self.is_available():
            return
        missing = [name for name, ok in self.available_modules().items() if not ok]
        raise OptionalQuantBackendUnavailable(
            f"{self.name} backend is not available; missing {', '.join(missing)}. {self.install_hint}"
        )


def unavailable_quantize(spec: QuantBackendSpec, *args, **kwargs):
    spec.require_available()
    raise NotImplementedError(
        f"{spec.name} dependency imports are available, but packed production quantization is not wired in v1. "
        "Use the fake backend for CI/export smoke or add a backend-specific implementation."
    )

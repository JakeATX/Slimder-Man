from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifest_schema import CompressionManifest


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def load_manifest(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CompressionManifest.model_validate(data).model_dump(mode="json")


def validate_manifest_dict(data: dict[str, Any]) -> CompressionManifest:
    return CompressionManifest.model_validate(data)

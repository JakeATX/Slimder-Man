from __future__ import annotations

from pathlib import Path
from typing import Any

from slimder_man.utils.hashing import sha256_file
from slimder_man.utils.json import write_json


EXPORT_ARTIFACT_NAMES = {
    "model.pt",
    "model.safetensors",
    "pytorch_model.bin",
    "config.json",
    "fake_quant_manifest.json",
    "quantization_manifest.json",
}


def collect_export_hashes(output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    hashes: dict[str, str] = {}
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        if path.name in EXPORT_ARTIFACT_NAMES or path.name.startswith("model-"):
            hashes[path.relative_to(out).as_posix()] = sha256_file(path)
    return hashes


def write_quant_export_manifest(
    output_dir: str | Path,
    backend: str,
    backend_manifest: dict[str, Any],
    source_checkpoint: str | None = None,
    export_format: str = "dequantized_hf_or_torch",
) -> dict[str, Any]:
    out = Path(output_dir)
    manifest = {
        "schema_version": "1.0",
        "backend": backend,
        "export_format": export_format,
        "source_checkpoint": source_checkpoint,
        "backend_manifest": backend_manifest,
        "artifact_hashes": collect_export_hashes(out),
        "note": "Export manifest describes stored artifacts; fake backend stores dequantized tensors for portability.",
    }
    write_json(out / "quant_export_manifest.json", manifest)
    return manifest

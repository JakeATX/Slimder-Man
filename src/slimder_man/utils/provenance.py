from __future__ import annotations

import hashlib
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

from slimder_man.config.schema import SlimderConfig


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def normalized_config_sha256(cfg: SlimderConfig) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version() -> str:
    try:
        return metadata.version("slimder-man")
    except metadata.PackageNotFoundError:
        return "unknown"


def git_commit(repo_root: str | Path = ".") -> str | None:
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def config_provenance(cfg: SlimderConfig, source_config_path: str | Path | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "normalized_config_sha256": normalized_config_sha256(cfg),
        "git_commit": git_commit(repo_root()),
        "package_version": package_version(),
    }
    if source_config_path is not None:
        config_path = Path(source_config_path).resolve()
        result["source_config_path"] = str(config_path)
        result["source_config_sha256"] = file_sha256(config_path)
    return result

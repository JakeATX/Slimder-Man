from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import redact_secret


@dataclass
class LocalPlan:
    commands: list[str]
    preflight: list[dict[str, str]]
    output_dir: str


def _quote_arg(value: str | Path) -> str:
    text = str(value)
    return '"' + text.replace('"', '\\"') + '"'


def local_preflight(cfg: SlimderConfig, repo_root: str | Path = ".") -> list[dict[str, str]]:
    root = Path(repo_root).resolve()
    checks = [
        {
            "name": "python",
            "status": "ok",
            "message": sys.version.split()[0],
        },
        {
            "name": "package",
            "status": "ok" if (root / "pyproject.toml").exists() else "warning",
            "message": str(root / "pyproject.toml"),
        },
        {
            "name": "config_output_dir",
            "status": "ok",
            "message": cfg.project.output_dir,
        },
    ]
    if shutil.which("nvidia-smi"):
        checks.append({"name": "cuda", "status": "ok", "message": "nvidia-smi available"})
    else:
        checks.append({"name": "cuda", "status": "warning", "message": "CUDA not detected; CPU/local smoke path only"})
    if cfg.teacher.load_mode == "transformers" and cfg.teacher.model_id_or_path != "dummy-hf-moe":
        checks.append(
            {
                "name": "full_model_local_run",
                "status": "warning",
                "message": "Use explicit analyze/compress/distill or remote launch for real full-model checkpoints",
            }
        )
    return checks


def local_dry_run_commands(config_path: str | Path, cfg: SlimderConfig) -> LocalPlan:
    config = Path(config_path)
    config_arg = _quote_arg(config)
    output_dir = _quote_arg(Path(cfg.project.output_dir) / "training" / "final")
    if cfg.teacher.load_mode == "tiny" or cfg.teacher.model_id_or_path == "dummy-hf-moe":
        commands = [f"python -m slimder_man.cli run {config_arg} --json"]
    else:
        commands = [
            f"python -m slimder_man.cli analyze {config_arg} --json",
            f"python -m slimder_man.cli recommend --config {config_arg} --preset {cfg.compression.preset} --write-config {config_arg} --json",
            f"python -m slimder_man.cli compress --config {config_arg} --stage 1 --json",
            f"python -m slimder_man.cli distill {config_arg} --stage 1 --json",
            f"python -m slimder_man.cli eval --checkpoint {output_dir} --json",
        ]
    return LocalPlan(
        commands=[redact_secret(command) for command in commands],
        preflight=local_preflight(cfg),
        output_dir=cfg.project.output_dir,
    )

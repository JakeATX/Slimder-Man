from __future__ import annotations

from pathlib import Path
import re

import yaml

from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import redact_secret


def _safe_remote_name(value: str | Path, fallback: str = "run") -> str:
    name = Path(str(value)).name or fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return safe or fallback


def skypilot_yaml(config_path: str | Path, cfg: SlimderConfig) -> str:
    source_config = "configs/source_config.yaml"
    remote_config = "configs/launch_config.yaml"
    remote_run = f"outputs/{_safe_remote_name(cfg.project.output_dir, 'run')}"
    resources = {
        "accelerators": cfg.runtime.skypilot.accelerators,
        "disk_size": 512,
        "use_spot": False,
        "autostop": {
            "idle_minutes": cfg.runtime.skypilot.autostop_minutes,
            "wait_for": "none",
        },
    }
    if cfg.runtime.skypilot.cloud != "auto":
        resources["infra"] = cfg.runtime.skypilot.cloud
    data = {
        "name": cfg.runtime.skypilot.cluster_name,
        "resources": resources,
        "workdir": ".",
        "envs": {
            "HF_TOKEN": "${HF_TOKEN}",
            "WANDB_API_KEY": "${WANDB_API_KEY}",
        },
        "file_mounts": {
            source_config: str(config_path),
        },
        "setup": "\n".join([
            "python -m pip install --upgrade pip",
            "python -m pip install -e .[dev]",
            f"python -m slimder_man.orchestration.materialize_config {source_config} {remote_config} --output-dir {remote_run} --json",
            f"python -m slimder_man.cli run {remote_config} --dry-run --json",
        ]),
        "run": "\n".join([
            "set -euo pipefail",
            "mkdir -p logs",
            f"python -m slimder_man.cli analyze {remote_config} --json | tee logs/analyze.log",
            f"python -m slimder_man.cli recommend --config {remote_config} --preset {cfg.compression.preset} --json | tee logs/recommend.log",
            f"python -m slimder_man.cli compress --config {remote_config} --stage 1 --json | tee logs/compress.log",
            f"python -m slimder_man.cli distill {remote_config} --stage 1 --json | tee logs/distill.log",
            f"python -m slimder_man.cli eval --checkpoint {remote_run}/training/final --json | tee logs/eval.log",
            f"echo 'Artifacts are under {remote_run}; use sky rsync-down for retrieval.'",
        ]),
    }
    return redact_secret(yaml.safe_dump(data, sort_keys=False))

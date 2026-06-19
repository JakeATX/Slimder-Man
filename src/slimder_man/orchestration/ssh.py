from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import redact_secret


@dataclass
class DryRun:
    commands: list[str]


def _quote(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _double_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _safe_remote_name(value: str | Path, fallback: str = "run") -> str:
    name = Path(str(value)).name or fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return safe or fallback


def _rsync_ssh_options(port: int, key_path: str | None) -> str:
    ssh_parts = ["ssh", "-p", str(port)]
    if key_path:
        ssh_parts.extend(["-i", _quote(key_path)])
    return f"-e {_double_quote(' '.join(ssh_parts))}"


def ssh_dry_run_commands(config_path: str | Path, cfg: SlimderConfig) -> DryRun:
    ssh = cfg.runtime.ssh
    host = ssh.host or "example.invalid"
    user = ssh.user or "user"
    target = f"{user}@{host}"
    ssh_parts = ["ssh"]
    if ssh.key_path:
        ssh_parts.extend(["-i", str(ssh.key_path).replace("\\", "/")])
    ssh_parts.extend(["-p", str(ssh.port), target])
    ssh_base = " ".join(_quote(part) if " " in part else part for part in ssh_parts)
    rsync_ssh = _rsync_ssh_options(ssh.port, ssh.key_path)
    remote_root = "~/slimder-man"
    source_config = f"{remote_root}/configs/{_safe_remote_name(config_path, 'config.yaml')}"
    remote_config = f"{remote_root}/configs/launch_config.yaml"
    remote_run = f"{remote_root}/outputs/{_safe_remote_name(cfg.project.output_dir, 'run')}"
    local_config = Path(config_path)
    local_output = Path(cfg.project.output_dir)
    cmds = [
        f"rsync -az --delete {rsync_ssh} --exclude .git --exclude .venv --exclude runs ./ {target}:{remote_root}/",
        f"{ssh_base} 'mkdir -p {remote_root}/configs {remote_root}/outputs {remote_root}/logs'",
        f"rsync -az {rsync_ssh} {_quote(local_config)} {target}:{source_config}",
        f"{ssh_base} 'cd {remote_root} && python --version && python -m pip --version'",
        f"{ssh_base} 'cd {remote_root} && nvidia-smi || echo CUDA_NOT_AVAILABLE'",
        f"{ssh_base} 'cd {remote_root} && python -m pip install -e .[dev]'",
        f"{ssh_base} 'cd {remote_root} && python -m slimder_man.orchestration.materialize_config {source_config} {remote_config} --output-dir {remote_run} --json'",
        f"{ssh_base} 'cd {remote_root} && test -f {remote_config} && python -m slimder_man.cli run {remote_config} --dry-run --json'",
        f"{ssh_base} 'cd {remote_root} && mkdir -p logs && nohup python -m slimder_man.cli analyze {remote_config} --json > logs/analyze.log 2>&1'",
        f"{ssh_base} 'cd {remote_root} && nohup python -m slimder_man.cli recommend --config {remote_config} --preset {cfg.compression.preset} --json > logs/recommend.log 2>&1'",
        f"{ssh_base} 'cd {remote_root} && nohup python -m slimder_man.cli compress --config {remote_config} --stage 1 --json > logs/compress.log 2>&1'",
        f"{ssh_base} 'cd {remote_root} && nohup python -m slimder_man.cli distill {remote_config} --stage 1 --json > logs/distill.log 2>&1'",
        f"{ssh_base} 'cd {remote_root} && nohup python -m slimder_man.cli eval --checkpoint {remote_run}/training/final --json > logs/eval.log 2>&1'",
        f"{ssh_base} 'tail -n 200 -f {remote_root}/logs/*.log'",
        f"{ssh_base} 'pkill -f slimder_man.cli || true'",
        f"rsync -az {rsync_ssh} {target}:{remote_run}/ {_quote(local_output)}/",
    ]
    return DryRun([redact_secret(c) for c in cmds])

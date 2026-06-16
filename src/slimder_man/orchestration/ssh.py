from __future__ import annotations

from dataclasses import dataclass

from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import redact_secret


@dataclass
class DryRun:
    commands: list[str]


def ssh_dry_run_commands(cfg: SlimderConfig) -> DryRun:
    ssh = cfg.runtime.ssh
    host = ssh.host or "example.invalid"
    user = ssh.user or "user"
    target = f"{user}@{host}"
    cmds = [
        f"rsync -az --exclude .venv ./ {target}:~/slimder-man/",
        f"ssh -p {ssh.port} {target} 'cd ~/slimder-man && nvidia-smi'",
        f"ssh -p {ssh.port} {target} 'cd ~/slimder-man && python -m slimder_man.cli run --config config.yaml'",
        f"rsync -az {target}:~/slimder-man/{cfg.project.output_dir}/ {cfg.project.output_dir}/",
    ]
    return DryRun([redact_secret(c) for c in cmds])

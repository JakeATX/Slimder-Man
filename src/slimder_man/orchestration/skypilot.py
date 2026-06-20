from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml

from slimder_man.config.schema import SlimderConfig
from slimder_man.orchestration.jobs import CommandExecutor, CommandResult, SubprocessExecutor
from slimder_man.orchestration.sync import shell_quote
from slimder_man.utils.hashing import redact_secret


def _safe_remote_name(value: str | Path, fallback: str = "run") -> str:
    name = Path(str(value)).name or fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return safe or fallback


@dataclass
class SkyPilotPlan:
    yaml: str
    task_path: str
    launch_command: str
    logs_command: str
    stop_command: str
    sync_command: str


@dataclass
class SkyPilotRunResult:
    backend: str
    dry_run: bool
    status: str
    commands: list[str]
    results: list[CommandResult]
    task_path: str
    failed_command: str | None = None


def skypilot_yaml(config_path: str | Path, cfg: SlimderConfig) -> str:
    source_config = "configs/source_config.yaml"
    remote_config = "configs/launch_config.yaml"
    remote_run = f"outputs/{_safe_remote_name(cfg.project.output_dir, 'run')}"
    resources = {
        "accelerators": cfg.runtime.skypilot.accelerators,
        "disk_size": cfg.runtime.skypilot.disk_size_gb,
        "use_spot": False,
        "autostop": {
            "idle_minutes": cfg.runtime.skypilot.autostop_minutes,
            "wait_for": "none",
        },
    }
    if cfg.runtime.skypilot.cloud != "auto":
        resources["infra"] = cfg.runtime.skypilot.cloud
    if cfg.runtime.skypilot.region:
        resources["region"] = cfg.runtime.skypilot.region
    if cfg.runtime.skypilot.image_id:
        resources["image_id"] = cfg.runtime.skypilot.image_id
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


def skypilot_plan(config_path: str | Path, cfg: SlimderConfig, task_path: str | Path | None = None) -> SkyPilotPlan:
    task = Path(task_path or Path(cfg.project.output_dir) / "skypilot_task.yaml")
    cluster = cfg.runtime.skypilot.cluster_name
    output_dir = Path(cfg.project.output_dir)
    yaml_text = skypilot_yaml(config_path, cfg)
    return SkyPilotPlan(
        yaml=yaml_text,
        task_path=str(task),
        launch_command=redact_secret(f"sky launch -c {cluster} {shell_quote(task)} --yes"),
        logs_command=redact_secret(f"sky logs {cluster} --follow"),
        stop_command=redact_secret(f"sky stop {cluster} --yes"),
        sync_command=redact_secret(
            f"sky rsync-down {cluster} outputs/{_safe_remote_name(cfg.project.output_dir, 'run')}/ {shell_quote(output_dir)}/"
        ),
    )


class SkyPilotRunner:
    def __init__(
        self,
        config_path: str | Path,
        cfg: SlimderConfig,
        executor: CommandExecutor | None = None,
        task_path: str | Path | None = None,
    ) -> None:
        self.config_path = config_path
        self.cfg = cfg
        self.executor = executor or SubprocessExecutor()
        self.task_path = task_path

    def plan(self) -> SkyPilotPlan:
        return skypilot_plan(self.config_path, self.cfg, self.task_path)

    def launch(self, dry_run: bool | None = None) -> SkyPilotRunResult:
        plan = self.plan()
        commands = [plan.launch_command]
        should_dry_run = self.cfg.runtime.skypilot.dry_run if dry_run is None else dry_run
        if should_dry_run:
            return SkyPilotRunResult("skypilot", True, "dry_run", commands, [], plan.task_path)
        task = Path(plan.task_path)
        task.parent.mkdir(parents=True, exist_ok=True)
        task.write_text(plan.yaml, encoding="utf-8")
        result = self.executor.run(plan.launch_command).redacted()
        status = "succeeded" if result.ok else "failed"
        return SkyPilotRunResult(
            "skypilot",
            False,
            status,
            commands,
            [result],
            plan.task_path,
            failed_command=None if result.ok else plan.launch_command,
        )

    def stream_logs(self):
        command = self.plan().logs_command
        stream = getattr(self.executor, "stream", None)
        if stream is not None:
            return (redact_secret(line) for line in stream(command))
        result = self.executor.run(command).redacted()
        return iter((result.stdout + result.stderr).splitlines())

    def stop(self) -> CommandResult:
        return self.executor.run(self.plan().stop_command).redacted()

    def sync_outputs(self) -> CommandResult:
        return self.executor.run(self.plan().sync_command).redacted()

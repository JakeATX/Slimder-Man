from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from slimder_man.config.schema import SlimderConfig
from slimder_man.orchestration.jobs import CommandExecutor, CommandResult, SubprocessExecutor
from slimder_man.orchestration.preflight import ssh_preflight_probes
from slimder_man.orchestration.sync import rsync_download_command, rsync_upload_command
from slimder_man.orchestration.sync import shell_quote as _quote
from slimder_man.utils.hashing import redact_secret


@dataclass
class DryRun:
    commands: list[str]


@dataclass
class SSHCommand:
    label: str
    command: str
    execute_in_launch: bool = True


@dataclass
class SSHRunResult:
    backend: str
    dry_run: bool
    status: str
    commands: list[str]
    results: list[CommandResult]
    failed_command: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"dry_run", "succeeded"}


def _safe_remote_name(value: str | Path, fallback: str = "run") -> str:
    name = Path(str(value)).name or fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return safe or fallback


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ssh_call(ssh_base: str, command: str) -> str:
    return f"{ssh_base} {_quote(command)}"


def _guarded_preflight_command(ssh_base: str, probe_command: str, actionable_failure: str) -> str:
    return _ssh_call(ssh_base, f"{probe_command} || (echo {_quote(actionable_failure)} && exit 1)")


def ssh_command_plan(
    config_path: str | Path,
    cfg: SlimderConfig,
    repo_root: str | Path | None = None,
) -> list[SSHCommand]:
    ssh = cfg.runtime.ssh
    host = ssh.host or "example.invalid"
    user = ssh.user or "user"
    target = f"{user}@{host}"
    ssh_parts = ["ssh"]
    if ssh.key_path:
        ssh_parts.extend(["-i", str(ssh.key_path).replace("\\", "/")])
    ssh_parts.extend(["-p", str(ssh.port), target])
    ssh_base = " ".join(_quote(part) if " " in part else part for part in ssh_parts)
    remote_root = "~/slimder-man"
    source_config = f"{remote_root}/configs/{_safe_remote_name(config_path, 'config.yaml')}"
    remote_config = f"{remote_root}/configs/launch_config.yaml"
    remote_run = f"{remote_root}/outputs/{_safe_remote_name(cfg.project.output_dir, 'run')}"
    local_config = Path(config_path)
    local_output = Path(cfg.project.output_dir)
    local_repo = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    local_repo_contents = local_repo.as_posix().rstrip("/") + "/"
    probes = ssh_preflight_probes(remote_root)
    steps = [
        SSHCommand(
            "sync_code",
            rsync_upload_command(
                local_repo_contents,
                f"{target}:{remote_root}/",
                port=ssh.port,
                key_path=ssh.key_path,
                delete=True,
                excludes=[".git", ".venv", "runs"],
            ),
        ),
        SSHCommand("mkdirs", _ssh_call(ssh_base, f"mkdir -p {remote_root}/configs {remote_root}/outputs {remote_root}/logs")),
        SSHCommand(
            "sync_config",
            rsync_upload_command(local_config, f"{target}:{source_config}", port=ssh.port, key_path=ssh.key_path),
        ),
    ]
    steps.extend(
        SSHCommand(
            f"preflight_{probe.name}",
            _guarded_preflight_command(ssh_base, probe.command, probe.actionable_failure),
        )
        for probe in probes
        if probe.name != "torch"
    )
    steps.extend([
        SSHCommand("install", _ssh_call(ssh_base, f"cd {remote_root} && python -m pip install -e .[dev]")),
    ])
    steps.extend(
        SSHCommand(
            f"preflight_{probe.name}",
            _guarded_preflight_command(ssh_base, probe.command, probe.actionable_failure),
        )
        for probe in probes
        if probe.name == "torch"
    )
    steps.extend([
        SSHCommand(
            "materialize_config",
            _ssh_call(
                ssh_base,
                f"cd {remote_root} && python -m slimder_man.orchestration.materialize_config {source_config} {remote_config} --output-dir {remote_run} --json",
            ),
        ),
        SSHCommand(
            "dry_run_gate",
            _ssh_call(
                ssh_base,
                f"cd {remote_root} && test -f {remote_config} && python -m slimder_man.cli run {remote_config} --dry-run --json",
            ),
        ),
        SSHCommand(
            "analyze",
            _ssh_call(
                ssh_base,
                f"cd {remote_root} && mkdir -p logs && python -m slimder_man.cli analyze {remote_config} --json > logs/analyze.log 2>&1",
            ),
        ),
        SSHCommand(
            "recommend",
            _ssh_call(
                ssh_base,
                f"cd {remote_root} && python -m slimder_man.cli recommend --config {remote_config} --preset {cfg.compression.preset} --write-config {remote_config} --json > logs/recommend.log 2>&1",
            ),
        ),
        SSHCommand(
            "compress",
            _ssh_call(
                ssh_base,
                f"cd {remote_root} && python -m slimder_man.cli compress --config {remote_config} --stage 1 --json > logs/compress.log 2>&1",
            ),
        ),
        SSHCommand(
            "distill",
            _ssh_call(
                ssh_base,
                f"cd {remote_root} && python -m slimder_man.cli distill {remote_config} --stage 1 --json > logs/distill.log 2>&1",
            ),
        ),
        SSHCommand(
            "eval",
            _ssh_call(
                ssh_base,
                f"cd {remote_root} && python -m slimder_man.cli eval --checkpoint {remote_run}/training/final --json > logs/eval.log 2>&1",
            ),
        ),
        SSHCommand("logs", _ssh_call(ssh_base, f"tail -n 200 -f {remote_root}/logs/*.log"), execute_in_launch=False),
        SSHCommand("stop", _ssh_call(ssh_base, "pkill -f slimder_man.cli || true"), execute_in_launch=False),
        SSHCommand(
            "sync_outputs",
            rsync_download_command(f"{target}:{remote_run}/", local_output, port=ssh.port, key_path=ssh.key_path),
        ),
    ])
    return steps


def ssh_dry_run_commands(config_path: str | Path, cfg: SlimderConfig) -> DryRun:
    cmds = [step.command for step in ssh_command_plan(config_path, cfg)]
    return DryRun([redact_secret(c) for c in cmds])


class SSHRunner:
    def __init__(
        self,
        config_path: str | Path,
        cfg: SlimderConfig,
        executor: CommandExecutor | None = None,
        repo_root: str | Path | None = None,
    ) -> None:
        self.config_path = config_path
        self.cfg = cfg
        self.executor = executor or SubprocessExecutor()
        self.repo_root = repo_root

    def plan(self) -> list[SSHCommand]:
        return ssh_command_plan(self.config_path, self.cfg, repo_root=self.repo_root)

    def launch(self, dry_run: bool | None = None) -> SSHRunResult:
        steps = self.plan()
        executable = [step for step in steps if step.execute_in_launch]
        commands = [redact_secret(step.command) for step in executable]
        should_dry_run = self.cfg.runtime.ssh.dry_run if dry_run is None else dry_run
        if should_dry_run:
            return SSHRunResult(backend="ssh", dry_run=True, status="dry_run", commands=commands, results=[])

        results: list[CommandResult] = []
        for step in executable:
            result = self.executor.run(step.command).redacted()
            results.append(result)
            if not result.ok:
                return SSHRunResult(
                    backend="ssh",
                    dry_run=False,
                    status="failed",
                    commands=commands[: len(results)],
                    results=results,
                    failed_command=redact_secret(step.command),
                )
        return SSHRunResult(backend="ssh", dry_run=False, status="succeeded", commands=commands, results=results)

    def stream_logs(self):
        command = next(step.command for step in self.plan() if step.label == "logs")
        stream = getattr(self.executor, "stream", None)
        if stream is not None:
            return (redact_secret(line) for line in stream(command))
        result = self.executor.run(command).redacted()
        lines = (result.stdout + result.stderr).splitlines()
        return iter(lines)

    def stop(self) -> CommandResult:
        command = next(step.command for step in self.plan() if step.label == "stop")
        return self.executor.run(command).redacted()

    def sync_outputs(self) -> CommandResult:
        command = next(step.command for step in self.plan() if step.label == "sync_outputs")
        return self.executor.run(command).redacted()

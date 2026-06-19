import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig, save_config
from slimder_man.orchestration.jobs import CommandResult
from slimder_man.orchestration.local import local_dry_run_commands, local_preflight
from slimder_man.orchestration.materialize_config import materialize_remote_config
from slimder_man.orchestration.skypilot import skypilot_yaml
from slimder_man.orchestration.ssh import SSHRunner, ssh_command_plan, ssh_dry_run_commands
from slimder_man.utils.hashing import redact_secret


class RecordingExecutor:
    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.commands: list[str] = []

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        if self.fail_on and self.fail_on in command:
            return CommandResult(command=command, returncode=17, stderr="boom")
        return CommandResult(command=command, returncode=0, stdout="ok")

    def stream(self, command: str):
        self.commands.append(command)
        yield "log-line-1"
        yield "token=hf_abcdef123"


def test_ssh_and_skypilot_dry_runs_redact(tmp_path: Path):
    config_path = tmp_path / "custom launch.yaml"
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "run out")},
        runtime={
            "backend": "ssh",
            "ssh": {"host": "host", "user": "user", "port": 2222, "key_path": str(tmp_path / "id key")},
            "skypilot": {"cloud": "aws", "autostop_minutes": 45},
        },
    )
    save_config(cfg, config_path)
    cmds = ssh_dry_run_commands(config_path, cfg).commands
    joined = "\n".join(cmds)
    assert any("rsync" in c for c in cmds)
    assert any("nvidia-smi" in c for c in cmds)
    assert "CUDA_NOT_AVAILABLE" not in joined
    assert any("df -h ." in c for c in cmds)
    assert any("import torch" in c for c in cmds)
    assert "TORCH_NOT_INSTALLED_YET" not in joined
    assert any("pip install -e .[dev]" in c for c in cmds)
    assert "custom-launch.yaml" in joined
    assert "configs/launch_config.yaml" in joined
    assert "outputs/run-out" in joined
    assert "mkdir -p ~/slimder-man/configs ~/slimder-man/outputs ~/slimder-man/logs" in joined
    assert "-e \"ssh -p 2222 -i '" in joined
    assert "/id key'\"" in joined
    assert "run out/training/final" not in joined
    assert any("cli analyze" in c for c in cmds)
    assert any("cli compress" in c for c in cmds)
    assert any("cli distill" in c for c in cmds)
    assert any("tail -n 200 -f" in c for c in cmds)
    assert any("pkill -f slimder_man.cli" in c for c in cmds)
    yml = skypilot_yaml(config_path, cfg)
    yml_data = yaml.safe_load(yml)
    assert yml_data["resources"]["infra"] == "aws"
    assert yml_data["resources"]["autostop"]["idle_minutes"] == 45
    assert "output_sync" not in yml_data and "workdir_sync" not in yml_data
    assert "accelerators" in yml and "slimder_man.cli run" in yml
    assert yml_data["file_mounts"]["configs/source_config.yaml"] == str(config_path)
    assert "configs/launch_config.yaml" in yml
    assert "outputs/run-out" in yml
    assert "slimder_man.cli analyze" in yml
    assert "slimder_man.cli compress" in yml
    assert "slimder_man.cli distill" in yml
    assert "HF_TOKEN" in yml
    assert "hf_***REDACTED***" in redact_secret("token=hf_abcdef123")
    assert "hf_dummy.yaml" in redact_secret("src/slimder_man/config/examples/hf_dummy.yaml")


def test_ssh_plan_syncs_resolved_repo_root_and_quotes_excludes(tmp_path: Path):
    cfg = SlimderConfig(project={"paper_faithful": False, "output_dir": str(tmp_path / "run")})
    repo_root = tmp_path / "repo root"
    config_path = tmp_path / "launch.yaml"
    save_config(cfg, config_path)

    first = ssh_command_plan(config_path, cfg, repo_root=repo_root)[0].command

    synced_source = str(repo_root.resolve()).replace("\\", "/").rstrip("/") + "/"
    assert synced_source in first
    assert " './'" not in first
    assert "--exclude '.git'" in first
    assert "--exclude '.venv'" in first


def test_ssh_runner_stops_on_failed_required_preflight(tmp_path: Path):
    config_path = tmp_path / "launch.yaml"
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "run")},
        runtime={"ssh": {"host": "host", "user": "user", "dry_run": False}},
    )
    save_config(cfg, config_path)
    executor = RecordingExecutor(fail_on="nvidia-smi")

    result = SSHRunner(config_path, cfg, executor=executor).launch()

    assert result.status == "failed"
    assert result.failed_command is not None and "nvidia-smi" in result.failed_command
    assert any("nvidia-smi" in command for command in executor.commands)
    assert not any("pip install -e" in command for command in executor.commands)


def test_ssh_runner_executes_ordered_stages_and_stops_on_failure(tmp_path: Path):
    config_path = tmp_path / "launch.yaml"
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "run")},
        runtime={"ssh": {"host": "host", "user": "user", "dry_run": False}},
    )
    save_config(cfg, config_path)
    executor = RecordingExecutor(fail_on="cli compress")

    result = SSHRunner(config_path, cfg, executor=executor).launch()

    assert result.status == "failed"
    assert result.failed_command is not None and "cli compress" in result.failed_command
    assert any("materialize_config" in command for command in executor.commands)
    assert any("cli analyze" in command for command in executor.commands)
    assert any("cli recommend" in command for command in executor.commands)
    assert any("cli compress" in command for command in executor.commands)
    assert not any("cli distill" in command for command in executor.commands)
    assert not any("tail -n 200 -f" in command for command in executor.commands)


def test_ssh_runner_exposes_log_stop_and_sync_operations(tmp_path: Path):
    config_path = tmp_path / "launch.yaml"
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "run")},
        runtime={"ssh": {"host": "host", "user": "user", "dry_run": False}},
    )
    save_config(cfg, config_path)
    executor = RecordingExecutor()
    runner = SSHRunner(config_path, cfg, executor=executor)

    logs = list(runner.stream_logs())
    stop = runner.stop()
    sync = runner.sync_outputs()

    assert logs == ["log-line-1", "token=hf_***REDACTED***"]
    assert "tail -n 200 -f" in executor.commands[0]
    assert stop.ok and "pkill -f slimder_man.cli" in executor.commands[1]
    assert sync.ok and executor.commands[2].startswith("rsync -az")


def test_local_launch_for_hf_dummy_emits_executable_run_plan(tmp_path: Path):
    config_path = Path("src/slimder_man/config/examples/hf_dummy.yaml").resolve()
    result = CliRunner().invoke(app, ["launch", str(config_path), "--backend", "local", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"] == "local"
    assert payload["output_dir"] == "runs/hf_dummy_moe_smoke"
    assert payload["commands"] == [f'python -m slimder_man.cli run "{config_path}" --json']
    assert any(check["name"] == "python" and check["status"] == "ok" for check in payload["preflight"])
    assert any(check["name"] == "cuda" for check in payload["preflight"])


def test_local_launch_for_real_transformer_emits_staged_plan_without_run_gate(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "out with space")},
        teacher={"load_mode": "transformers", "model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"},
    )
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    plan = local_dry_run_commands(config_path, cfg)

    assert len(plan.commands) == 5
    assert "cli analyze" in plan.commands[0]
    assert "cli recommend --config" in plan.commands[1]
    assert "cli compress --config" in plan.commands[2]
    assert "cli distill" in plan.commands[3]
    assert "cli eval --checkpoint" in plan.commands[4]
    assert f'"{tmp_path / "out with space" / "training" / "final"}"' in plan.commands[4]
    assert not any(" cli run " in command for command in plan.commands)
    assert any(check["name"] == "full_model_local_run" and check["status"] == "warning" for check in plan.preflight)


def test_local_preflight_reports_missing_package_warning(tmp_path: Path):
    cfg = SlimderConfig()
    checks = local_preflight(cfg, repo_root=tmp_path)
    package = next(check for check in checks if check["name"] == "package")
    assert package["status"] == "warning"


def test_materialize_remote_config_rewrites_output_dir(tmp_path: Path):
    source = tmp_path / "source.yaml"
    dest = tmp_path / "remote" / "launch_config.yaml"
    cfg = SlimderConfig(project={"paper_faithful": False, "output_dir": str(tmp_path / "local out")})
    save_config(cfg, source)

    payload = materialize_remote_config(source, dest, "outputs/remote-out")
    rewritten = SlimderConfig.model_validate(yaml.safe_load(dest.read_text(encoding="utf-8")))

    assert payload["destination"] == str(dest)
    assert rewritten.project.output_dir == "outputs/remote-out"

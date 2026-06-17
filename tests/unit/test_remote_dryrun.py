import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig
from slimder_man.orchestration.local import local_dry_run_commands, local_preflight
from slimder_man.orchestration.skypilot import skypilot_yaml
from slimder_man.orchestration.ssh import ssh_dry_run_commands
from slimder_man.utils.hashing import redact_secret


def test_ssh_and_skypilot_dry_runs_redact():
    cfg = SlimderConfig(project={"paper_faithful": False}, runtime={"backend": "ssh", "ssh": {"host": "host", "user": "user"}})
    cmds = ssh_dry_run_commands(cfg).commands
    assert any("rsync" in c for c in cmds)
    assert any("nvidia-smi" in c for c in cmds)
    yml = skypilot_yaml(cfg)
    assert "accelerators" in yml and "slimder run" in yml
    assert "hf_***REDACTED***" in redact_secret("token=hf_abcdef123")
    assert "hf_dummy.yaml" in redact_secret("src/slimder_man/config/examples/hf_dummy.yaml")


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

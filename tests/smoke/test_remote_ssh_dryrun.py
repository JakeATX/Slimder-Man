from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig, save_config


def test_remote_ssh_dryrun_cli(tmp_path):
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "run with space")},
        runtime={"backend": "ssh", "ssh": {"host": "host", "user": "user", "port": 2200, "key_path": str(tmp_path / "id_ed25519")}},
    )
    config = tmp_path / "bespoke config.yaml"
    save_config(cfg, config)
    result = CliRunner().invoke(app, ["launch", str(config), "--backend", "ssh", "--json"])
    assert result.exit_code == 0, result.output
    assert "rsync" in result.output
    assert "mkdir -p ~/slimder-man/configs ~/slimder-man/outputs ~/slimder-man/logs" in result.output
    assert "-e \\\"ssh -p 2200 -i '" in result.output
    assert "bespoke-config.yaml" in result.output
    assert "configs/launch_config.yaml" in result.output
    assert "outputs/run-with-space" in result.output
    assert "run with space/training/final" not in result.output
    assert "nvidia-smi" in result.output
    assert "pip install -e .[dev]" in result.output
    assert "slimder_man.cli analyze" in result.output
    assert "slimder_man.cli compress" in result.output
    assert "slimder_man.cli distill" in result.output
    assert "tail -n 200 -f" in result.output

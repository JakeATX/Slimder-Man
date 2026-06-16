from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig, save_config


def test_remote_ssh_dryrun_cli(tmp_path):
    cfg = SlimderConfig(project={"paper_faithful": False, "output_dir": str(tmp_path / "run")}, runtime={"backend": "ssh", "ssh": {"host": "host", "user": "user"}})
    config = tmp_path / "config.yaml"
    save_config(cfg, config)
    result = CliRunner().invoke(app, ["launch", str(config), "--backend", "ssh", "--json"])
    assert result.exit_code == 0, result.output
    assert "rsync" in result.output
    assert "nvidia-smi" in result.output

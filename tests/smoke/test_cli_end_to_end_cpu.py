from pathlib import Path

from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig, save_config


def test_cli_end_to_end_cpu(tmp_path: Path):
    cfg = SlimderConfig(project={"output_dir": str(tmp_path / "run")})
    config_path = tmp_path / "config.yaml"
    save_config(cfg, config_path)
    runner = CliRunner()
    result = runner.invoke(app, ["run", str(config_path), "--json"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "run" / "analysis" / "analysis_report.md").exists()
    ckpt = tmp_path / "run" / "checkpoints" / "stage_1_compressed"
    assert (ckpt / "compression_manifest.json").exists()
    assert (tmp_path / "run" / "training" / "final" / "model.pt").exists()
    val = runner.invoke(app, ["validate-checkpoint", str(ckpt), "--json"])
    assert val.exit_code == 0, val.output
    eval_result = runner.invoke(app, ["eval", str(tmp_path / "run" / "training" / "final"), "--json"])
    assert eval_result.exit_code == 0, eval_result.output

import json
from pathlib import Path

from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig, save_config


def test_cli_run_executes_progressive_tiny_stages(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "run")},
        progressive={"schedule": "depth_first", "stages": 2, "token_split": [0.5, 0.5]},
        calibration={"sample_count": 4, "sequence_length": 8},
        training={"token_budget": 32, "train_steps": 1, "warmup_steps": 0},
        compression={"target": {"hidden_size": 12, "remove_last_n_layers": 2, "routed_experts": 4, "routed_top_k": 2}},
    )
    config_path = tmp_path / "config.yaml"
    save_config(cfg, config_path)

    result = CliRunner().invoke(app, ["run", str(config_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    stages = payload["progressive"]["stages"]
    assert [stage["stage"] for stage in stages] == [1, 2]
    assert [stage["tokens"] for stage in stages] == [16, 16]
    assert stages[0]["manifest"]["target"]["hidden_size"] == 16
    assert stages[0]["manifest"]["target"]["remove_last_n_layers"] == 1
    assert stages[0]["manifest"]["target"]["routed_experts"] == 8
    assert stages[1]["manifest"]["target"]["hidden_size"] == 12
    assert stages[1]["manifest"]["target"]["remove_last_n_layers"] == 2
    assert stages[1]["manifest"]["target"]["routed_experts"] == 4
    assert (tmp_path / "run" / "progressive" / "stage_1" / "compressed" / "compression_manifest.json").exists()
    assert (tmp_path / "run" / "progressive" / "stage_2" / "training" / "final" / "model.pt").exists()
    assert (tmp_path / "run" / "run_summary.json").exists()

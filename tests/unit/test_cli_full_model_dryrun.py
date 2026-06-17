import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig


def test_run_dryrun_accepts_transformers_config_without_loading_model(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "out")},
        teacher={
            "load_mode": "transformers",
            "model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "dtype": "bfloat16",
            "device_map": "auto",
        },
    )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config_path), "--dry-run", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "dry_run"
    assert payload["teacher"]["load_mode"] == "transformers"
    assert payload["teacher"]["model_id_or_path"] == "Qwen/Qwen3-Next-80B-A3B-Instruct"
    assert payload["stages"][0]["compress"] is True

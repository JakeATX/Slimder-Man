import json
from pathlib import Path

from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.compression.manifests import load_manifest
from slimder_man.config.schema import SlimderConfig, save_config
from slimder_man.utils.hashing import sha256_file


def test_cli_run_executes_progressive_tiny_stages(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "run")},
        progressive={"schedule": "depth_first", "stages": 2, "token_split": [0.5, 0.5]},
        calibration={"sample_count": 4, "sequence_length": 8},
        training={"token_budget": 32, "train_steps": 1, "warmup_steps": 2},
        compression={"target": {"hidden_size": 12, "remove_last_n_layers": 2, "routed_experts": 4, "routed_top_k": 2}},
    )
    config_path = tmp_path / "config.yaml"
    save_config(cfg, config_path)

    result = CliRunner().invoke(app, ["run", str(config_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    stages = payload["progressive"]["stages"]
    assert payload["progressive"]["global_total_steps"] == 2
    assert [stage["stage"] for stage in stages] == [1, 2]
    assert [stage["tokens"] for stage in stages] == [16, 16]
    assert [stage["train_steps"] for stage in stages] == [1, 1]
    assert [stage["global_step_start"] for stage in stages] == [0, 1]
    assert [stage["global_step_end"] for stage in stages] == [1, 2]
    assert stages[0]["training"]["logs"][0]["step"] == 1
    assert stages[1]["training"]["logs"][0]["step"] == 2
    assert stages[1]["training"]["logs"][0]["lr"] > stages[0]["training"]["logs"][0]["lr"]
    assert stages[0]["manifest"]["target"]["hidden_size"] == 16
    assert stages[0]["manifest"]["target"]["remove_last_n_layers"] == 1
    assert stages[0]["manifest"]["target"]["routed_experts"] == 8
    assert stages[1]["manifest"]["target"]["hidden_size"] == 12
    assert stages[1]["manifest"]["target"]["remove_last_n_layers"] == 2
    assert stages[1]["manifest"]["target"]["routed_experts"] == 4
    stage_1_manifest_path = tmp_path / "run" / "progressive" / "stage_1" / "compressed" / "compression_manifest.json"
    stage_2_manifest_path = tmp_path / "run" / "progressive" / "stage_2" / "compressed" / "compression_manifest.json"
    assert stage_1_manifest_path.exists()
    stage_1_manifest = load_manifest(stage_1_manifest_path)
    stage_2_manifest = load_manifest(stage_2_manifest_path)
    assert stage_1_manifest["stage_provenance"]["stage"] == 1
    assert stage_1_manifest["stage_provenance"]["total_stages"] == 2
    assert stage_1_manifest["stage_provenance"]["stage_token_budget"] == 16
    assert stage_1_manifest["stage_provenance"]["source"] == "teacher"
    assert stage_1_manifest["stage_provenance"]["previous_checkpoint"] is None
    assert stage_2_manifest["stage_provenance"]["stage"] == 2
    assert stage_2_manifest["stage_provenance"]["source"] == "previous_stage_checkpoint"
    assert Path(stage_2_manifest["stage_provenance"]["previous_checkpoint"]).parts[-2:] == ("stage_1", "compressed")
    assert stage_2_manifest["stage_provenance"]["final_stage"] is True
    for stage_manifest in (stage_1_manifest, stage_2_manifest):
        calibration = stage_manifest["calibration_artifacts"]
        assert calibration["manifest_sha256"] == sha256_file(calibration["manifest_path"])
        first_layer = stage_manifest["experts"]["layers"][0]
        assert Path(first_layer["score_artifact"]["path"]).exists()
        assert Path(first_layer["similarity_artifact"]["path"]).exists()
        assert stage_manifest["provenance"]["source_config_sha256"] == sha256_file(config_path)
    assert (tmp_path / "run" / "progressive" / "stage_2" / "training" / "final" / "model.pt").exists()
    assert (tmp_path / "run" / "run_summary.json").exists()

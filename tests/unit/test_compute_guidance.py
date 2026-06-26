import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig, load_config, save_config
from slimder_man.orchestration.compute_guidance import compute_guidance, compute_guidance_markdown


def qwen36_config(tmp_path: Path) -> SlimderConfig:
    return SlimderConfig(
        project={"paper_faithful": True, "output_dir": str(tmp_path / "qwen36")},
        teacher={"load_mode": "transformers", "model_id_or_path": "Qwen/Qwen3.6-35B-A3B", "dtype": "bfloat16"},
        runtime={"backend": "skypilot", "skypilot": {"accelerators": "H100:4"}},
        training={"sequence_length": 4096},
        compression={"target": {"hidden_size": 1536, "remove_last_n_layers": 10, "routed_experts": 128, "routed_top_k": 6}},
    )


def test_qwen36_compute_guidance_profile_and_memory(tmp_path: Path):
    cfg = qwen36_config(tmp_path)

    guidance = compute_guidance(cfg)

    assert guidance["model"]["model_id"] == "Qwen/Qwen3.6-35B-A3B"
    assert guidance["model"]["total_params_b"] == 35.0
    assert guidance["model"]["active_params_b"] == 3.0
    assert guidance["model"]["layers"] == 40
    assert guidance["model"]["routed_experts"] == 256
    assert guidance["model"]["top_k"] == 8
    assert guidance["memory_estimates_gb"]["teacher_weights_fp16_or_bf16"] == 70.0
    assert guidance["local"]["status"] == "not_recommended_for_full_framework"
    assert guidance["remote"]["status"] == "recommended"
    assert guidance["api"]["status"] == "full_logits_required"
    assert "remote full-logit worker" in guidance["api"]["summary"] or "full-vocabulary teacher logits" in guidance["api"]["summary"]


def test_compute_guidance_markdown_is_user_actionable(tmp_path: Path):
    markdown = compute_guidance_markdown(compute_guidance(qwen36_config(tmp_path)))

    assert "Compute guidance for Qwen/Qwen3.6-35B-A3B" in markdown
    assert "Teacher bf16/fp16 weights" in markdown
    assert "SkyPilot" in markdown
    assert "full-logit" in markdown


def test_compute_guidance_cli_and_dry_run_include_guidance(tmp_path: Path):
    config_path = tmp_path / "qwen36.yaml"
    save_config(qwen36_config(tmp_path), config_path)
    runner = CliRunner()

    guidance_result = runner.invoke(app, ["compute-guidance", str(config_path), "--json"])
    dry_run_result = runner.invoke(app, ["run", str(config_path), "--dry-run", "--json"])
    launch_result = runner.invoke(app, ["launch", str(config_path), "--backend", "skypilot", "--json"])

    assert guidance_result.exit_code == 0, guidance_result.output
    assert dry_run_result.exit_code == 0, dry_run_result.output
    assert launch_result.exit_code == 0, launch_result.output
    guidance = json.loads(guidance_result.output)
    dry_run = json.loads(dry_run_result.output)
    launch = json.loads(launch_result.output)
    assert guidance["model"]["family"] == "qwen3.6-a3b"
    assert dry_run["compute_guidance"]["model"]["family"] == "qwen3.6-a3b"
    assert launch["compute_guidance"]["remote"]["status"] == "recommended"


def test_qwen36_example_config_parses_and_defaults_to_remote_dry_run():
    cfg = load_config("src/slimder_man/config/examples/qwen36_35b_a3b_remote.yaml")

    assert cfg.teacher.model_id_or_path == "Qwen/Qwen3.6-35B-A3B"
    assert cfg.teacher.load_mode == "transformers"
    assert cfg.project.paper_faithful is True
    assert cfg.runtime.backend == "skypilot"
    assert cfg.runtime.local.allow_full_model_run is False
    assert cfg.runtime.skypilot.dry_run is True
    assert cfg.training.allow_smoke_trainer is False
    assert compute_guidance(cfg)["recommended_path"].startswith("Start locally with config-only")


def test_unknown_architecture_guidance_uses_architecture_fallback(tmp_path: Path):
    cfg = SlimderConfig(project={"paper_faithful": False}, teacher={"load_mode": "transformers", "model_id_or_path": "local/custom"})
    arch = {
        "model_type": "custom_moe",
        "total_params": 12_000_000_000,
        "hidden_size": 1024,
        "num_layers": 24,
        "moe_layers": [{"num_routed_experts": 64, "top_k": 4, "num_shared_experts": 1}],
    }

    guidance = compute_guidance(cfg, architecture=arch)

    assert guidance["model"]["source"] == "architecture_or_config"
    assert guidance["model"]["total_params_b"] == 12.0
    assert guidance["model"]["routed_experts"] == 64
    assert guidance["api"]["status"] == "augmented_allowed"

import json
import yaml

from slimder_man.config.schema import SlimderConfig
from slimder_man.ui.app import artifact_index, build_config_yaml, config_warnings, log_tail, paper_faithful_quant_state, run_cli_with_yaml, run_ui_command


def test_ui_config_generation_includes_teacher_dataset_and_runtime_fields():
    data = yaml.safe_load(
        build_config_yaml(
            project_name="custom",
            paper_faithful=False,
            quantization=True,
            teacher_model_id_or_path="Qwen/Qwen3-Next-80B-A3B-Instruct",
            teacher_load_mode="transformers",
            teacher_dtype="bfloat16",
            teacher_revision="main",
            trust_remote_code=False,
            dataset_type="jsonl",
            dataset_path="data/train.jsonl",
            dataset_split="validation",
            text_field="prompt",
            sample_count=32,
            sequence_length=128,
            token_budget=2048,
            train_steps=7,
            runtime_backend="ssh",
            local_num_gpus="2",
            ssh_host="gpu.example",
            ssh_user="trainer",
            ssh_port=2222,
            ssh_dry_run=False,
            skypilot_cluster_name="slimder-prod",
            skypilot_accelerators="A100:4",
            skypilot_cloud="aws",
            tracking_backend="none",
            compression_preset="aggressive_80",
        )
    )
    cfg = SlimderConfig.model_validate(data)
    assert cfg.project.name == "custom"
    assert cfg.project.output_dir == "runs/custom"
    assert cfg.teacher.model_id_or_path == "Qwen/Qwen3-Next-80B-A3B-Instruct"
    assert cfg.teacher.load_mode == "transformers"
    assert cfg.teacher.dtype == "bfloat16"
    assert cfg.teacher.revision == "main"
    assert cfg.teacher.trust_remote_code is False
    assert cfg.calibration.dataset.type == "jsonl"
    assert cfg.calibration.dataset.path == "data/train.jsonl"
    assert cfg.calibration.dataset.split == "validation"
    assert cfg.calibration.dataset.text_field == "prompt"
    assert cfg.calibration.sample_count == 32
    assert cfg.training.sequence_length == 128
    assert cfg.training.token_budget == 2048
    assert cfg.training.train_steps == 7
    assert cfg.compression.preset == "aggressive_80"
    assert cfg.runtime.backend == "ssh"
    assert cfg.runtime.local.num_gpus == 2
    assert cfg.runtime.ssh.host == "gpu.example"
    assert cfg.runtime.ssh.user == "trainer"
    assert cfg.runtime.ssh.port == 2222
    assert cfg.runtime.ssh.dry_run is False
    assert cfg.runtime.skypilot.cluster_name == "slimder-prod"
    assert cfg.runtime.skypilot.accelerators == "A100:4"
    assert cfg.runtime.skypilot.cloud == "aws"
    assert cfg.runtime.tracking.backend == "none"


def test_ui_paper_faithful_forces_augmented_quantization_off():
    faithful = SlimderConfig.model_validate(yaml.safe_load(build_config_yaml(paper_faithful=True, quantization=True)))
    augmented = SlimderConfig.model_validate(yaml.safe_load(build_config_yaml(paper_faithful=False, quantization=True)))

    assert faithful.project.paper_faithful is True
    assert faithful.quantization.enabled is False
    assert augmented.project.paper_faithful is False
    assert augmented.quantization.enabled is True
    assert paper_faithful_quant_state(True) == {"value": False, "interactive": False}
    assert paper_faithful_quant_state(False) == {"interactive": True}


def test_ui_helpers_run_tiny_analyze_recommend_and_run(tmp_path):
    yaml_text = build_config_yaml(
        project_name="ui_tiny",
        sample_count=4,
        sequence_length=8,
        token_budget=32,
        train_steps=1,
        output_dir=str(tmp_path / "ui_tiny"),
    )

    analyze = json.loads(run_cli_with_yaml(yaml_text, "analyze"))
    recommend = json.loads(run_cli_with_yaml(yaml_text, "recommend", preset="balanced_50"))
    run = json.loads(run_cli_with_yaml(yaml_text, "run"))
    launch = json.loads(run_cli_with_yaml(yaml_text, "launch", backend="local"))
    eval_result = json.loads(run_cli_with_yaml(yaml_text, "eval", checkpoint=str(tmp_path / "ui_tiny" / "training" / "final")))

    assert analyze["analysis_dir"] == str(tmp_path / "ui_tiny" / "analysis")
    assert recommend["preset"] == "balanced_50"
    assert recommend["candidates"]
    assert launch["backend"] == "local"
    assert launch["commands"]
    assert run["perplexity"] > 0
    assert eval_result["checkpoint"] == str(tmp_path / "ui_tiny" / "training" / "final")
    assert eval_result["perplexity"] > 0
    assert (tmp_path / "ui_tiny" / "run_summary.json").exists()
    artifacts = artifact_index(str(tmp_path / "ui_tiny"))
    logs = log_tail(str(tmp_path / "ui_tiny"))
    assert "run_summary.json" in artifacts
    assert "training_report.md" in artifacts
    assert "training_report.md" in logs
    assert "No warnings." == config_warnings(yaml_text)


def test_ui_command_uses_current_fields_without_generated_yaml(monkeypatch):
    captured = {}

    def fake_run(yaml_text: str, command: str, preset: str = "balanced_50", **kwargs) -> str:
        captured["yaml"] = yaml_text
        captured["command"] = command
        captured["preset"] = preset
        captured["kwargs"] = kwargs
        return "{}"

    monkeypatch.setattr("slimder_man.ui.app.run_cli_with_yaml", fake_run)

    result = run_ui_command(
        "recommend",
        "current_fields_project",
        False,
        True,
        "tiny",
        "tiny",
        "float32",
        "",
        True,
        "synthetic",
        "",
        "",
        "train",
        "text",
        6,
        7,
        128,
        1,
        "local",
        "auto",
        "",
        "",
        22,
        True,
        "slimder",
        "H100:8",
        "auto",
        "none",
        preset="all",
        compression_preset="extreme_90",
    )

    cfg = SlimderConfig.model_validate(yaml.safe_load(captured["yaml"]))
    assert result == "{}"
    assert captured["command"] == "recommend"
    assert captured["preset"] == "all"
    assert captured["kwargs"]["stage"] == 1
    assert captured["kwargs"]["backend"] == "local"
    assert cfg.project.name == "current_fields_project"
    assert cfg.project.paper_faithful is False
    assert cfg.quantization.enabled is True
    assert cfg.compression.preset == "extreme_90"
    assert cfg.calibration.sample_count == 6
    assert cfg.calibration.sequence_length == 7


def test_ui_warnings_surface_high_risk_and_full_model_contract(tmp_path):
    yaml_text = build_config_yaml(
        project_name="warn",
        paper_faithful=False,
        teacher_model_id_or_path="Qwen/Qwen3-Next-80B-A3B-Instruct",
        teacher_load_mode="transformers",
        output_dir=str(tmp_path / "warn"),
        compression_preset="aggressive_80",
    )
    warning_text = config_warnings(yaml_text)

    assert "aggressive_80" in warning_text
    assert "Non-paper-faithful" in warning_text
    assert "allow_full_model_run=true" in warning_text

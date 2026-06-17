import yaml

from slimder_man.config.schema import SlimderConfig
from slimder_man.ui.app import build_config_yaml


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

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from slimder_man.config.schema import SlimderConfig, save_config


def _entrypoint_config(tmp_path: Path) -> Path:
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "run")},
        calibration={"sample_count": 4, "sequence_length": 8},
        training={
            "train_steps": 2,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "sequence_length": 8,
            "warmup_steps": 0,
        },
    )
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    return path


def test_train_loop_module_entrypoint_runs_tiny_cpu(tmp_path: Path):
    config_path = _entrypoint_config(tmp_path)
    out_dir = tmp_path / "module_training"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "slimder_man.distill.train_loop",
            "--config",
            str(config_path),
            "--output-dir",
            str(out_dir),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["entrypoint"] == "slimder_man.distill.train_loop"
    assert payload["mode"] == "tiny"
    assert payload["global_step"] == 2
    assert payload["gradient_accumulation_steps"] == 2
    assert (out_dir / "final" / "model.pt").exists()
    assert (out_dir / "training_report.md").exists()


def test_train_loop_rejects_arbitrary_transformers_entrypoint(tmp_path: Path):
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "run")},
        teacher={"load_mode": "transformers", "model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"},
    )
    config_path = tmp_path / "qwen.yaml"
    save_config(cfg, config_path)

    proc = subprocess.run(
        [sys.executable, "-m", "slimder_man.distill.train_loop", "--config", str(config_path), "--json"],
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "dummy-hf-moe" in proc.stderr


def test_accelerate_launch_train_loop_tiny_cpu(tmp_path: Path):
    config_path = _entrypoint_config(tmp_path)
    out_dir = tmp_path / "accelerate_training"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes=1",
            "-m",
            "slimder_man.distill.train_loop",
            "--config",
            str(config_path),
            "--output-dir",
            str(out_dir),
            "--json",
        ],
        text=True,
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0 and "No module named accelerate" in proc.stderr:
        pytest.skip("accelerate is not installed")

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout[proc.stdout.find("{") :])
    assert payload["global_step"] == 2
    assert payload["mode"] == "tiny"
    assert (out_dir / "final" / "model.pt").exists()

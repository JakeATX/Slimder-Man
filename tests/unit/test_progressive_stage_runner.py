from pathlib import Path

import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.calibration.collectors import CalibrationResult
from slimder_man.compression.manifests import load_manifest
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.stage_runner import run_tiny_progressive_stages
from slimder_man.utils.hashing import sha256_file


def _calibration() -> CalibrationResult:
    return CalibrationResult(
        hidden_scores=torch.ones(16),
        per_layer_hidden_scores=[torch.ones(16) for _ in range(9)],
        expert_frequency=[torch.arange(8, dtype=torch.float32) for _ in range(4)],
        expert_soft=[torch.arange(8, dtype=torch.float32) for _ in range(4)],
        expert_reap=[torch.arange(8, dtype=torch.float32) for _ in range(4)],
        expert_similarity=[torch.eye(8) for _ in range(4)],
    )


def test_progressive_stage_runner_wires_stage_specific_configs(tmp_path):
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path)},
        progressive={"schedule": "depth_first", "stages": 2, "token_split": [0.25, 0.75]},
        training={"token_budget": 100, "train_steps": 0, "global_batch_size": 10, "micro_batch_size": 5, "sequence_length": 5},
        compression={"target": {"hidden_size": 12, "remove_last_n_layers": 2, "routed_experts": 4, "routed_top_k": 2}},
    )
    teacher = TinyMoEForCausalLM()
    seen = []

    def calibrate_fn(model, stage_cfg):
        seen.append(("calibrate", stage_cfg.compression.target.remove_last_n_layers, stage_cfg.training.token_budget))
        return _calibration()

    def compress_fn(model, stage_cfg, cal, output_dir):
        seen.append(
            (
                "compress",
                stage_cfg.compression.target.remove_last_n_layers,
                stage_cfg.compression.target.hidden_size,
                stage_cfg.compression.target.routed_experts,
                str(output_dir),
            )
        )
        return model, {"target": stage_cfg.compression.target.model_dump()}

    def train_fn(model, student, stage_cfg, output_dir):
        seen.append(("train", stage_cfg.training.token_budget, stage_cfg.training.train_steps, str(output_dir)))
        return {"global_step": stage_cfg.training.train_steps}

    result = run_tiny_progressive_stages(teacher, cfg, tmp_path / "progressive", calibrate_fn, compress_fn, train_fn)

    assert [stage["tokens"] for stage in result["stages"]] == [25, 75]
    assert ("compress", 1, 16, 8, str(tmp_path / "progressive" / "stage_1" / "compressed")) in seen
    assert ("compress", 2, 12, 4, str(tmp_path / "progressive" / "stage_2" / "compressed")) in seen
    assert result["global_total_steps"] == 3
    assert ("train", 25, 1, str(tmp_path / "progressive" / "stage_1" / "training")) in seen
    assert ("train", 75, 2, str(tmp_path / "progressive" / "stage_2" / "training")) in seen
    assert [stage["train_steps"] for stage in result["stages"]] == [1, 2]


def test_progressive_stage_runner_executes_real_tiny_stages(tmp_path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        progressive={"schedule": "depth_first", "stages": 2, "token_split": [0.5, 0.5]},
        calibration={"sample_count": 4, "sequence_length": 8},
        training={"token_budget": 20, "train_steps": 1, "warmup_steps": 0},
        compression={"target": {"hidden_size": 12, "remove_last_n_layers": 2, "routed_experts": 4, "routed_top_k": 2}},
    )
    teacher = TinyMoEForCausalLM()

    result = run_tiny_progressive_stages(teacher, cfg, tmp_path / "real_progressive")

    assert len(result["stages"]) == 2
    assert (tmp_path / "real_progressive" / "stage_1" / "compressed" / "compression_manifest.json").exists()
    assert (tmp_path / "real_progressive" / "stage_2" / "compressed" / "compression_manifest.json").exists()
    stage_1_manifest = load_manifest(tmp_path / "real_progressive" / "stage_1" / "compressed" / "compression_manifest.json")
    stage_2_manifest = load_manifest(tmp_path / "real_progressive" / "stage_2" / "compressed" / "compression_manifest.json")
    assert stage_1_manifest["calibration_artifacts"]["manifest_sha256"] == sha256_file(stage_1_manifest["calibration_artifacts"]["manifest_path"])
    assert stage_2_manifest["calibration_artifacts"]["manifest_sha256"] == sha256_file(stage_2_manifest["calibration_artifacts"]["manifest_path"])
    assert stage_1_manifest["experts"]["layers"][0]["score_artifact"]["tensor"] == "soft_logits"
    assert stage_2_manifest["experts"]["layers"][0]["similarity_artifact"]["metric"] == "router_weights"
    assert stage_1_manifest["stage_provenance"]["stage"] == 1
    assert Path(stage_2_manifest["stage_provenance"]["previous_checkpoint"]).parts[-2:] == ("stage_1", "compressed")
    assert (tmp_path / "real_progressive" / "stage_2" / "training" / "training_report.md").exists()
    assert result["global_total_steps"] == 2
    assert result["stages"][0]["training"]["global_step"] == 1
    assert result["stages"][1]["training"]["global_step"] == 2
    assert result["stages"][1]["global_step_start"] == 1

from pathlib import Path

import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.calibration.collectors import CalibrationResult
from slimder_man.compression.manifests import load_manifest
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.stage_runner import run_generic_progressive_stages, run_tiny_progressive_stages
from tests.fixtures.hf_dummy_moe import DummyHfMoeConfig, DummyHfMoeForCausalLM
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


def _hf_calibration(hidden_size: int = 32, layers: int = 3, experts: int = 6) -> CalibrationResult:
    return CalibrationResult(
        hidden_scores=torch.arange(1, hidden_size + 1, dtype=torch.float32),
        per_layer_hidden_scores=[torch.arange(1, hidden_size + 1, dtype=torch.float32) for _ in range(layers)],
        expert_frequency=[torch.arange(1, experts + 1, dtype=torch.float32) for _ in range(layers)],
        expert_soft=[torch.arange(1, experts + 1, dtype=torch.float32) for _ in range(layers)],
        expert_reap=[torch.arange(1, experts + 1, dtype=torch.float32) for _ in range(layers)],
        expert_similarity=[torch.eye(experts) for _ in range(layers)],
        router_logits_similarity=[torch.eye(experts) for _ in range(layers)],
        router_weights_similarity=[torch.eye(experts) for _ in range(layers)],
        expert_layer_indices=list(range(layers)),
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
    assert ("compress", 1, 12, 4, str(tmp_path / "progressive" / "stage_2" / "compressed")) in seen
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
    assert Path(stage_2_manifest["stage_provenance"]["previous_checkpoint"]).parts[-2:] == ("training", "final")
    assert stage_2_manifest["target"]["remove_last_n_layers"] == 1
    assert stage_2_manifest["stage_provenance"]["cumulative_target"]["remove_last_n_layers"] == 2
    assert stage_2_manifest["stage_provenance"]["cumulative_target"]["layers"] == 2
    assert (tmp_path / "real_progressive" / "stage_2" / "training" / "training_report.md").exists()
    assert result["global_total_steps"] == 2
    assert result["stages"][0]["training"]["global_step"] == 1
    assert result["stages"][1]["training"]["global_step"] == 2
    assert result["stages"][1]["global_step_start"] == 1


def test_generic_progressive_stage_runner_wires_non_tiny_stage_chain(tmp_path):
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        progressive={"schedule": "depth_first", "stages": 2, "token_split": [0.5, 0.5]},
        training={"token_budget": 64, "train_steps": 0, "global_batch_size": 4, "micro_batch_size": 2, "sequence_length": 8},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 2, "routed_experts": 4, "routed_top_k": 2}},
    )
    teacher = DummyHfMoeForCausalLM()
    seen = []

    def calibrate_fn(model, stage_cfg):
        seen.append(("calibrate", model.config.num_hidden_layers, stage_cfg.compression.target.hidden_size))
        return _hf_calibration(model.config.hidden_size, model.config.num_hidden_layers, model.config.num_experts), {
            "type": "synthetic",
            "samples": [],
        }

    def compress_fn(model, stage_cfg, cal, adapter, output_dir, **kwargs):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        seen.append(
            (
                "compress",
                model.config.num_hidden_layers,
                stage_cfg.compression.target.remove_last_n_layers,
                stage_cfg.compression.target.hidden_size,
                stage_cfg.compression.target.routed_experts,
                kwargs["stage_provenance"]["source"],
            )
        )
        target_cfg = DummyHfMoeConfig(
            hidden_size=stage_cfg.compression.target.hidden_size,
            num_hidden_layers=model.config.num_hidden_layers - stage_cfg.compression.target.remove_last_n_layers,
            num_experts=stage_cfg.compression.target.routed_experts,
            num_experts_per_tok=stage_cfg.compression.target.routed_top_k,
        )
        student = DummyHfMoeForCausalLM(target_cfg)
        return student, {"target": stage_cfg.compression.target.model_dump(), "stage_provenance": kwargs["stage_provenance"]}

    def train_fn(teacher_model, student, stage_cfg, output_dir, batches, global_step_offset, global_total_steps, **_kwargs):
        checkpoint = Path(output_dir) / "final"
        checkpoint.mkdir(parents=True, exist_ok=True)
        seen.append(("train", student.config.num_hidden_layers, stage_cfg.training.train_steps, global_step_offset, global_total_steps))
        return {"checkpoint": str(checkpoint), "global_step": global_step_offset + stage_cfg.training.train_steps, "logs": []}

    def load_checkpoint_fn(path: Path):
        assert path.parts[-2:] == ("training", "final")
        last_compress = [item for item in seen if item[0] == "compress"][-1]
        return DummyHfMoeForCausalLM(
            DummyHfMoeConfig(
                hidden_size=last_compress[3],
                num_hidden_layers=last_compress[1] - last_compress[2],
                num_experts=last_compress[4],
                num_experts_per_tok=2,
            )
        )

    result = run_generic_progressive_stages(
        teacher,
        cfg,
        tmp_path / "generic_progressive",
        load_teacher_fn=lambda: DummyHfMoeForCausalLM(),
        load_checkpoint_fn=load_checkpoint_fn,
        calibrate_fn=calibrate_fn,
        compress_fn=compress_fn,
        train_fn=train_fn,
    )

    assert result["global_total_steps"] == 2
    assert [stage["global_step_start"] for stage in result["stages"]] == [0, 1]
    assert [stage["global_step_end"] for stage in result["stages"]] == [1, 2]
    assert ("compress", 3, 1, 32, 6, "teacher") in seen
    assert ("compress", 2, 1, 24, 4, "previous_stage_checkpoint") in seen
    assert ("train", 2, 1, 0, 2) in seen
    assert ("train", 1, 1, 1, 2) in seen
    assert result["stages"][1]["stage_provenance"]["previous_checkpoint"].endswith(str(Path("training") / "final"))
    assert result["stages"][1]["stage_provenance"]["cumulative_target"]["remove_last_n_layers"] == 2

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from typer.testing import CliRunner

from slimder_man.calibration.artifacts import write_calibration_artifacts
from slimder_man.calibration.collectors import collect_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.config.schema import CalibrationConfig, DatasetConfig, SlimderConfig, save_config
from slimder_man.utils.hashing import sha256_file

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyHfMoeForCausalLM, DummyTokenizer


def _text_cfg(path: Path, output_dir: Path, similarity_metric: str = "router_logits") -> SlimderConfig:
    return SlimderConfig(
        project={"output_dir": str(output_dir), "paper_faithful": False},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        calibration={
            "dataset": {"type": "text", "path": str(path)},
            "sample_count": 2,
            "sequence_length": 4,
            "seed": 2026,
        },
        compression={
            "target": {"hidden_size": 16, "routed_experts": 4, "routed_top_k": 2, "remove_last_n_layers": 0},
            "experts": {"similarity_metric": similarity_metric},
        },
    )


def test_calibration_artifact_writer_persists_tensors_and_provenance(tmp_path: Path):
    text_path = tmp_path / "calibration.txt"
    text_path.write_text("alpha beta gamma delta epsilon zeta", encoding="utf-8")
    cfg = _text_cfg(text_path, tmp_path / "runs")
    tokenizer = DummyTokenizer()
    model = DummyHfMoeForCausalLM()
    batches, source_manifest = sample_calibration_tokens(cfg.calibration, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)
    calibration = collect_calibration(model, batches)

    out_dir = tmp_path / "analysis"
    manifest = write_calibration_artifacts(out_dir, cfg, calibration, source_manifest, {"model_type": "dummy_hf_moe"})

    expected_files = {
        "hidden_importance.safetensors",
        "hidden_keep_indices_16.json",
        "routing_summary.json",
        "calibration_manifest.json",
    }
    expected_files.update({f"expert_stats_layer_{idx}.safetensors" for idx in range(model.config.num_hidden_layers)})
    for idx in range(model.config.num_hidden_layers):
        expected_files.update(
            {
                f"expert_similarity_layer_{idx}_router_logits.safetensors",
                f"expert_similarity_layer_{idx}_router_weights.safetensors",
                f"expert_similarity_layer_{idx}_expert_outputs.safetensors",
            }
        )
    assert expected_files.issubset({path.name for path in out_dir.iterdir()})

    hidden = load_file(out_dir / "hidden_importance.safetensors")
    assert hidden["global"].shape == (model.config.hidden_size,)
    assert hidden["layer_0"].shape == (model.config.hidden_size,)
    keep = json.loads((out_dir / "hidden_keep_indices_16.json").read_text(encoding="utf-8"))
    assert keep["target_hidden_size"] == 16
    assert keep["indices"] == sorted(keep["indices"])
    assert len(keep["indices"]) == 16

    stats = load_file(out_dir / "expert_stats_layer_0.safetensors")
    assert set(stats) == {"frequency", "soft_logits", "reap", "reap_assigned_token_mean"}
    assert stats["frequency"].shape == (model.config.num_experts,)
    assert torch.equal(stats["reap"], stats["reap_assigned_token_mean"])
    sim = load_file(out_dir / "expert_similarity_layer_0_router_logits.safetensors")["similarity"]
    assert sim.shape == (model.config.num_experts, model.config.num_experts)
    assert torch.allclose(sim, sim.T)

    written_manifest = json.loads((out_dir / "calibration_manifest.json").read_text(encoding="utf-8"))
    assert written_manifest == manifest
    assert written_manifest["calibration"]["sample_hashes"] == source_manifest["sample_hashes"]
    assert written_manifest["experts"]["reap_convention"] == "assigned_token_mean_gate_weighted_output_norm2"
    assert len(written_manifest["calibration"]["sample_hashes"]) == 2
    assert written_manifest["calibration"]["source"]["source_hash"]
    for artifact in written_manifest["artifacts"].values():
        assert artifact["sha256"] == sha256_file(out_dir / artifact["path"])


def test_generic_calibration_hooks_uninstrumented_moe_layers():
    model = DummyHfMoeForCausalLM()
    for layer in model.model.layers:
        moe = layer.mlp
        original_forward = moe.forward

        def clearing_forward(x, *, _original_forward=original_forward, _moe=moe):
            out = _original_forward(x)
            _moe.last_router_logits = None
            _moe.last_topk_indices = None
            _moe.last_topk_weights = None
            _moe.last_expert_output_norm2 = None
            return out

        moe.forward = clearing_forward
    batches, _ = sample_calibration_tokens(SlimderConfig(calibration={"sample_count": 2, "sequence_length": 8}).calibration, vocab_size=model.config.vocab_size)

    calibration = collect_calibration(model, batches)

    assert calibration.representation == "router_hook_recomputed_expert_outputs"
    assert len(calibration.expert_frequency) == model.config.num_hidden_layers
    assert calibration.expert_frequency[0].shape == (model.config.num_experts,)
    assert calibration.expert_soft[0].shape == (model.config.num_experts,)
    assert calibration.expert_reap[0].shape == (model.config.num_experts,)
    assert torch.isfinite(calibration.hidden_scores).all()
    assert torch.isfinite(calibration.expert_frequency[0]).all()
    assert torch.isfinite(calibration.expert_soft[0]).all()
    assert torch.isfinite(calibration.expert_reap[0]).all()
    assert calibration.expert_soft[0].sum() > 0
    assert calibration.expert_similarity[0].shape == (model.config.num_experts, model.config.num_experts)


def test_generic_calibration_restores_training_mode():
    model = DummyHfMoeForCausalLM()
    model.train()
    batches, _ = sample_calibration_tokens(SlimderConfig(calibration={"sample_count": 1, "sequence_length": 8}).calibration, vocab_size=model.config.vocab_size)

    collect_calibration(model, batches)

    assert model.training is True


def test_non_tiny_analyze_writes_calibration_artifacts(monkeypatch, tmp_path: Path):
    from slimder_man import cli

    text_path = tmp_path / "calibration.txt"
    text_path.write_text("alpha beta gamma delta epsilon zeta", encoding="utf-8")
    cfg = _text_cfg(text_path, tmp_path / "runs", similarity_metric="expert_outputs")
    config_path = tmp_path / "config.yaml"
    save_config(cfg, config_path)

    monkeypatch.setattr(cli, "_load_model", lambda _cfg: DummyHfMoeForCausalLM())
    monkeypatch.setattr(cli, "_load_tokenizer", lambda _cfg: DummyTokenizer())

    result = CliRunner().invoke(cli.app, ["analyze", str(config_path), "--json"])

    assert result.exit_code == 0, result.output
    analysis_dir = tmp_path / "runs" / "analysis"
    manifest = json.loads((analysis_dir / "calibration_manifest.json").read_text(encoding="utf-8"))
    routing = json.loads((analysis_dir / "routing_summary.json").read_text(encoding="utf-8"))
    assert manifest["teacher_load_mode"] == "transformers"
    assert manifest["experts"]["similarity_metric"] == "expert_outputs"
    assert manifest["calibration"]["tokenizer"] == "DummyTokenizer"
    assert routing["similarity_metric"] == "expert_outputs"
    assert routing["reap_convention"] == "assigned_token_mean_gate_weighted_output_norm2"
    assert (analysis_dir / "hidden_importance.safetensors").exists()
    assert (analysis_dir / "expert_similarity_layer_0_expert_outputs.safetensors").exists()

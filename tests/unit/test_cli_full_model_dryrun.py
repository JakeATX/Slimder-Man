import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM, DummyTokenizer
from slimder_man.analyze.config_only import describe_config_architecture
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.manifests import load_manifest
from slimder_man.config.schema import SlimderConfig
from slimder_man.eval.perplexity import causal_lm_perplexity
from slimder_man.utils.json import read_json
from slimder_man.utils.hashing import sha256_file


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
    assert len(payload["local"]["commands"]) == 5
    assert "cli analyze" in payload["local"]["commands"][0]
    assert "cli compress" in payload["local"]["commands"][2]
    assert "cli run" not in " ".join(payload["local"]["commands"])
    assert any(check["name"] == "full_model_local_run" for check in payload["local"]["preflight"])


def test_config_only_analyze_uses_transformers_config_without_loading_model(monkeypatch, tmp_path: Path):
    from slimder_man import cli

    class FakeQwen3NextConfig:
        model_type = "qwen3_next"
        hidden_size = 2048
        vocab_size = 151936
        num_hidden_layers = 48
        num_experts = 512
        shared_expert_intermediate_size = 512
        num_experts_per_tok = 10
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        tie_word_embeddings = False

    calls: list[str] = []

    def fail_if_called(name):
        def _fail(*_args, **_kwargs):
            calls.append(name)
            raise AssertionError(f"--config-only should not call {name}")

        return _fail

    monkeypatch.setattr(cli, "_load_model", fail_if_called("_load_model"))
    monkeypatch.setattr(cli, "_load_tokenizer", fail_if_called("_load_tokenizer"))
    monkeypatch.setattr(cli, "_load_transformers_tokenizer", fail_if_called("_load_transformers_tokenizer"))
    monkeypatch.setattr(cli, "sample_calibration_tokens", fail_if_called("sample_calibration_tokens"))
    monkeypatch.setattr(cli, "collect_calibration", fail_if_called("collect_calibration"))
    monkeypatch.setattr(cli, "collect_tiny_calibration", fail_if_called("collect_tiny_calibration"))
    monkeypatch.setattr(cli, "_load_transformers_config", lambda _cfg: FakeQwen3NextConfig())
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "out")},
        teacher={"load_mode": "transformers", "model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"},
    )
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["analyze", str(config_path), "--config-only", "--json"])

    assert result.exit_code == 0, result.output
    assert calls == []
    payload = json.loads(result.output)
    arch = payload["architecture"]
    assert payload["config_only"] is True
    assert arch["source"] == "config_only"
    assert arch["calibration_status"] == "not_run"
    assert arch["weights_loaded"] is False
    assert arch["parameter_count_status"] == "not_available_without_weights"
    assert arch["model_type"] == "qwen3_next"
    assert arch["hidden_size"] == 2048
    assert arch["vocab_size"] == 151936
    assert arch["num_layers"] == 48
    assert len(arch["block_kinds"]) == 48
    assert arch["num_linear_attention_layers"] == 36
    assert arch["num_full_attention_layers"] == 12
    assert len(arch["moe_layers"]) == 48
    assert arch["moe_layers"][0]["num_routed_experts"] == 512
    assert arch["moe_layers"][0]["num_shared_experts"] == 1
    assert arch["moe_layers"][0]["top_k"] == 10
    assert arch["has_mtp"] is False
    assert arch["mtp_depths"] == 0
    assert arch["tied_embeddings"] is False
    assert payload["analysis_dir"] == str(tmp_path / "out" / "analysis")
    assert (tmp_path / "out" / "analysis" / "architecture.json").exists()
    report = (tmp_path / "out" / "analysis" / "analysis_report.md").read_text(encoding="utf-8")
    assert "- source: config_only" in report
    assert "- calibration: not_run" in report
    assert "- weights_loaded: false" in report
    assert "- parameter_counts: not_checkpoint_derived" in report
    assert "- recommendation_estimates: formula_based_from_config" in report
    first_recommendation = payload["recommendations"][0]
    assert first_recommendation["target_total_param_reduction"] == 0.5
    assert first_recommendation["estimated_total_param_reduction"] == pytest.approx(0.5, abs=0.01)
    assert first_recommendation["reduction_solver"] == "estimated_param_grid"


def test_config_only_analyze_rejects_tiny_mode(tmp_path: Path):
    cfg = SlimderConfig(project={"output_dir": str(tmp_path / "out")}, teacher={"load_mode": "tiny"})
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["analyze", str(config_path), "--config-only", "--json"])

    assert result.exit_code == 2
    assert "Usage:" in result.output


def test_recommend_config_only_writes_applied_qwen_anchor_config(monkeypatch, tmp_path: Path):
    from slimder_man import cli

    class FakeQwen3NextConfig:
        model_type = "qwen3_next"
        hidden_size = 2048
        vocab_size = 151936
        num_hidden_layers = 48
        num_experts = 512
        shared_expert_intermediate_size = 512
        num_experts_per_tok = 10
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        tie_word_embeddings = False

    monkeypatch.setattr(cli, "_load_transformers_config", lambda _cfg: FakeQwen3NextConfig())
    monkeypatch.setattr(cli, "_load_model", lambda _cfg: (_ for _ in ()).throw(AssertionError("recommend --config-only loaded weights")))
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "out")},
        teacher={"load_mode": "transformers", "model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"},
    )
    config_path = tmp_path / "qwen.yaml"
    applied_path = tmp_path / "qwen.applied.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "recommend",
            "--config",
            str(config_path),
            "--preset",
            "slimqwen_anchor",
            "--candidate-id",
            "slimqwen_anchor_1",
            "--write-config",
            str(applied_path),
            "--config-only",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    applied = SlimderConfig.model_validate(yaml.safe_load(applied_path.read_text(encoding="utf-8")))
    assert payload["written_config"] == str(applied_path)
    assert payload["selected_plan"]["target"]["hidden_size"] == 1536
    assert applied.compression.target.hidden_size == 1536
    assert applied.compression.target.remove_last_n_layers == 12
    assert applied.compression.target.routed_experts == 256
    assert applied.compression.target.routed_top_k == 8
    assert applied.compression.plan is not None
    assert applied.compression.plan.candidate_id == "slimqwen_anchor_1"
    assert applied.compression.plan.source_architecture_fingerprint == payload["selected_plan"]["source_architecture_fingerprint"]


def test_config_only_architecture_repeats_short_block_patterns():
    class FakeConfig:
        model_type = "qwen3_next"
        hidden_size = 16
        vocab_size = 128
        num_hidden_layers = 5
        num_experts = 8
        shared_expert_intermediate_size = 32
        num_experts_per_tok = 2
        layer_types = ["linear_attention", "full_attention"]

    arch = describe_config_architecture(FakeConfig())

    assert arch["block_kinds"] == [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "full_attention",
        "linear_attention",
    ]
    assert len(arch["moe_layers"]) == 5
    assert arch["moe_layers"][0]["num_shared_experts"] == 1


def test_run_executes_checked_in_hf_dummy_pipeline_without_monkeypatch(tmp_path: Path):
    config_path = Path("src/slimder_man/config/examples/hf_dummy.yaml").resolve()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["run", str(config_path), "--json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        run_dir = Path("runs/hf_dummy_moe_smoke")
        ckpt = run_dir / "checkpoints" / "stage_1_compressed"
        summary = read_json(run_dir / "run_summary.json")
        assert summary == payload
        assert Path(payload["analysis"]) == run_dir / "analysis"
        assert payload["checkpoint"] == str(ckpt)
        assert payload["training"]["global_step"] == 1
        assert math.isfinite(payload["perplexity"]) and payload["perplexity"] > 0
        assert payload["recommendations"]
        assert payload["calibration_manifest"]["calibration"]["tokenizer"] == "DummyTokenizer"
        assert payload["calibration_manifest"]["calibration"]["source"]["type"] == "text"
        assert Path(payload["calibration_manifest"]["calibration"]["source"]["path"]).is_absolute()
        assert (run_dir / "analysis" / "architecture.json").exists()
        assert (run_dir / "analysis" / "calibration_manifest.json").exists()
        assert (run_dir / "analysis" / "analysis_report.md").exists()
        assert (ckpt / "model.safetensors").exists()
        assert (ckpt / "config.json").exists()
        assert (ckpt / "tokenizer_config.json").exists()
        assert (run_dir / "training" / "training_report.md").exists()
        assert (run_dir / "training" / "final" / "model.safetensors").exists()
        assert (run_dir / "run_summary.json").exists()
        manifest = load_manifest(ckpt / "compression_manifest.json")
        assert manifest["teacher_model"] == "dummy-hf-moe"
        assert manifest["provenance"]["normalized_config_sha256"]
        assert manifest["provenance"]["source_config_sha256"] == sha256_file(config_path)
        assert Path(manifest["provenance"]["source_config_path"]) == config_path
        expected_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert manifest["provenance"]["git_commit"] == expected_commit
        assert manifest["calibration_artifacts"]["manifest_sha256"] == sha256_file(run_dir / "analysis" / "calibration_manifest.json")
        assert manifest["calibration_artifacts"]["calibration"]["tokenizer_fingerprint"]["class"] == "DummyTokenizer"
        assert Path(manifest["experts"]["layers"][0]["score_artifact"]["path"]).exists()
        assert Path(manifest["experts"]["layers"][0]["similarity_artifact"]["path"]).exists()
        assert manifest["tokenizer"]["saved"] is True
        assert manifest["artifact_hashes"]["model.safetensors"] == sha256_file(ckpt / "model.safetensors")
        assert manifest["artifact_hashes"]["config.json"] == sha256_file(ckpt / "config.json")
        assert manifest["artifact_hashes"]["tokenizer_config.json"] == sha256_file(ckpt / "tokenizer_config.json")
        final_model = DummyHfMoeForCausalLM.from_pretrained(run_dir / "training" / "final")
        final_batches, final_manifest = sample_calibration_tokens(
            SlimderConfig(
                calibration={
                    "dataset": {"type": "text", "path": payload["calibration_manifest"]["calibration"]["source"]["path"]},
                    "sample_count": 1,
                    "sequence_length": 8,
                }
            ).calibration,
            vocab_size=final_model.config.vocab_size,
            tokenizer=DummyTokenizer(),
        )
        assert final_manifest["tokenizer"] == "DummyTokenizer"
        out = final_model(input_ids=final_batches[0], labels=final_batches[0])
        assert out.loss is not None
        assert math.isfinite(float(out.loss.detach()))
        final_ppl = causal_lm_perplexity(final_model, final_batches)
        assert math.isfinite(final_ppl) and final_ppl > 0


def test_checkpoint_commands_support_hf_dummy_outputs(tmp_path: Path):
    config_path = Path("src/slimder_man/config/examples/hf_dummy.yaml").resolve()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        run_result = runner.invoke(app, ["run", str(config_path), "--json"])
        assert run_result.exit_code == 0, run_result.output

        run_dir = Path("runs/hf_dummy_moe_smoke")
        compressed = run_dir / "checkpoints" / "stage_1_compressed"
        final = run_dir / "training" / "final"
        eval_result = runner.invoke(app, ["eval", "--checkpoint", str(final), "--json"])
        assert eval_result.exit_code == 0, eval_result.output
        eval_payload = json.loads(eval_result.output)
        assert eval_payload["kind"] == "dummy_hf_moe"
        assert math.isfinite(eval_payload["perplexity"]) and eval_payload["perplexity"] > 0

        validate_result = runner.invoke(app, ["validate-checkpoint", "--checkpoint", str(compressed), "--json"])
        assert validate_result.exit_code == 0, validate_result.output
        validate_payload = json.loads(validate_result.output)
        assert validate_payload["kind"] == "dummy_hf_moe"
        assert validate_payload["valid"] is True
        assert validate_payload["errors"] == []
        assert validate_payload["manifest"]["teacher_model"] == "dummy-hf-moe"
        assert validate_payload["manifest"]["calibration_artifacts"]["manifest_sha256"] == sha256_file(run_dir / "analysis" / "calibration_manifest.json")

        consolidated = Path("consolidated_hf")
        consolidate_result = runner.invoke(app, ["consolidate-checkpoint", "--checkpoint", str(compressed), "--out", str(consolidated), "--json"])
        assert consolidate_result.exit_code == 0, consolidate_result.output
        consolidate_payload = json.loads(consolidate_result.output)
        assert consolidate_payload["kind"] == "dummy_hf_moe"
        assert (consolidated / "model.safetensors").exists()
        assert (consolidated / "config.json").exists()
        assert (consolidated / "tokenizer_config.json").exists()
        assert (consolidated / "compression_manifest.json").exists()
        assert (consolidated / "calibration_artifacts" / "calibration_manifest.json").exists()
        assert consolidate_payload["artifact_hashes"]["model.safetensors"] == sha256_file(consolidated / "model.safetensors")
        assert consolidate_payload["artifact_hashes"]["tokenizer_config.json"] == sha256_file(consolidated / "tokenizer_config.json")
        consolidated_manifest = load_manifest(consolidated / "compression_manifest.json")
        assert consolidated_manifest["calibration_artifacts"]["manifest_sha256"] == sha256_file(consolidated / "calibration_artifacts" / "calibration_manifest.json")
        assert Path(consolidated_manifest["experts"]["layers"][0]["score_artifact"]["path"]).is_file()
        reloaded = DummyHfMoeForCausalLM.from_pretrained(consolidated)
        batches, _ = sample_calibration_tokens(SlimderConfig(calibration={"sample_count": 1, "sequence_length": 8}).calibration, vocab_size=reloaded.config.vocab_size)
        assert causal_lm_perplexity(reloaded, batches) > 0
        shutil.rmtree(run_dir)
        validate_consolidated = runner.invoke(app, ["validate-checkpoint", "--checkpoint", str(consolidated), "--json"])
        assert validate_consolidated.exit_code == 0, validate_consolidated.output
        assert json.loads(validate_consolidated.output)["valid"] is True


def test_validate_checkpoint_fails_on_tampered_calibration_artifact(tmp_path: Path):
    config_path = Path("src/slimder_man/config/examples/hf_dummy.yaml").resolve()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        run_result = runner.invoke(app, ["run", str(config_path), "--json"])
        assert run_result.exit_code == 0, run_result.output

        run_dir = Path("runs/hf_dummy_moe_smoke")
        compressed = run_dir / "checkpoints" / "stage_1_compressed"
        stats_path = run_dir / "analysis" / "expert_stats_layer_0.safetensors"
        stats_path.write_bytes(stats_path.read_bytes() + b"tamper")

        validate_result = runner.invoke(app, ["validate-checkpoint", "--checkpoint", str(compressed), "--json"])

        assert validate_result.exit_code == 0, validate_result.output
        payload = json.loads(validate_result.output)
        assert payload["valid"] is False
        assert any("calibration artifact hash mismatch" in error for error in payload["errors"])


def test_run_rejects_non_dummy_transformers_without_explicit_local_opt_in(monkeypatch, tmp_path: Path):
    from slimder_man import cli

    loaded = False

    def fail_if_loaded(_cfg):
        nonlocal loaded
        loaded = True
        raise AssertionError("non-opted-in full run should reject before model loading")

    monkeypatch.setattr(cli, "_load_model", fail_if_loaded)
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "out")},
        teacher={"load_mode": "transformers", "model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"},
    )
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config_path), "--json"])

    assert result.exit_code != 0
    assert "allow_full_model_run=true" in result.output
    assert loaded is False


def test_compress_rejects_arbitrary_transformers_without_applied_plan_before_loading(monkeypatch, tmp_path: Path):
    from slimder_man import cli

    loaded = False

    def fail_if_loaded(_cfg):
        nonlocal loaded
        loaded = True
        raise AssertionError("compress should reject missing compression.plan before loading weights")

    monkeypatch.setattr(cli, "_load_model", fail_if_loaded)
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "out")},
        teacher={"load_mode": "transformers", "model_id_or_path": "org/real-qwen"},
        compression={"target": {"hidden_size": 12, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
    )
    config_path = tmp_path / "real.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["compress", "--config", str(config_path), "--json"])

    assert result.exit_code != 0
    assert "compression.plan is required" in result.output
    assert loaded is False


def test_run_rejects_non_dummy_transformers_without_smoke_trainer_opt_in(monkeypatch, tmp_path: Path):
    from slimder_man import cli

    loaded = False

    def fail_if_loaded(_cfg):
        nonlocal loaded
        loaded = True
        raise AssertionError("disabled smoke trainer should reject before model loading")

    monkeypatch.setattr(cli, "_load_model", fail_if_loaded)
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "out")},
        teacher={"load_mode": "transformers", "model_id_or_path": "org/tiny-moe"},
        runtime={"local": {"allow_full_model_run": True}},
    )
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config_path), "--json"])

    assert result.exit_code != 0
    assert "allow_smoke_trainer=true" in result.output
    assert loaded is False


def test_run_accepts_opted_in_non_dummy_transformers_through_generic_hf_pipeline(monkeypatch, tmp_path: Path):
    from slimder_man import cli

    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "out")},
        teacher={"load_mode": "transformers", "model_id_or_path": "org/non-dummy-moe"},
        student={"output_format": "hf_safetensors"},
        runtime={"local": {"allow_full_model_run": True}},
        calibration={"sample_count": 2, "sequence_length": 8},
        training={"train_steps": 1, "micro_batch_size": 1, "global_batch_size": 1, "allow_smoke_trainer": True},
        compression={"target": {"hidden_size": 12, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
    )
    from slimder_man.analyze.plans import apply_recommendation_to_config
    from slimder_man.analyze.architecture import describe_model

    cfg, _ = apply_recommendation_to_config(cfg, describe_model(DummyHfMoeForCausalLM()), preset="balanced_50")
    config_path = tmp_path / "full.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(cli, "_load_model", lambda _cfg: DummyHfMoeForCausalLM())
    monkeypatch.setattr(cli, "_load_tokenizer", lambda _cfg: DummyTokenizer())

    result = CliRunner().invoke(app, ["run", str(config_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["manifest"]["teacher_model"] == "org/non-dummy-moe"
    assert Path(payload["checkpoint"], "model.safetensors").exists()
    assert payload["training"]["global_step"] == 1


def test_run_executes_progressive_dummy_hf_stages(tmp_path: Path):
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "progressive_hf")},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        student={"output_format": "hf_safetensors"},
        progressive={"schedule": "depth_first", "stages": 2, "token_split": [0.5, 0.5]},
        calibration={"sample_count": 2, "sequence_length": 8},
        training={"token_budget": 32, "train_steps": 1, "warmup_steps": 0, "micro_batch_size": 1, "global_batch_size": 1},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 2, "routed_experts": 4, "routed_top_k": 2}},
    )
    config_path = tmp_path / "progressive_hf.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config_path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    stages = payload["progressive"]["stages"]
    assert [stage["stage"] for stage in stages] == [1, 2]
    assert [stage["global_step_start"] for stage in stages] == [0, 1]
    assert [stage["global_step_end"] for stage in stages] == [1, 2]
    assert stages[0]["manifest"]["target"]["hidden_size"] == 32
    assert stages[0]["manifest"]["target"]["remove_last_n_layers"] == 1
    assert stages[1]["manifest"]["target"]["hidden_size"] == 24
    assert stages[1]["manifest"]["target"]["remove_last_n_layers"] == 1
    assert stages[1]["stage_provenance"]["previous_checkpoint"].endswith(str(Path("training") / "final"))
    assert payload["perplexity"] > 0


def test_causal_lm_perplexity_rejects_missing_losses():
    class NoLossModel:
        def eval(self):
            return self

        def __call__(self, input_ids, labels=None):
            return type("NoLossOutput", (), {"loss": None})()

    batch = sample_calibration_tokens(SlimderConfig(calibration={"sample_count": 1, "sequence_length": 4}).calibration)[0][0]
    try:
        causal_lm_perplexity(NoLossModel(), [batch])
    except ValueError as exc:
        assert "returned no losses" in str(exc)
    else:
        raise AssertionError("causal_lm_perplexity should reject models that return no loss")

    try:
        causal_lm_perplexity(NoLossModel(), [])
    except ValueError as exc:
        assert "without evaluation batches" in str(exc)
    else:
        raise AssertionError("causal_lm_perplexity should reject empty batches")


def test_causal_lm_perplexity_rejects_nonfinite_losses():
    class BadLossModel:
        def __init__(self, value: float):
            self.value = value

        def eval(self):
            return self

        def __call__(self, input_ids, labels=None):
            return type("BadLossOutput", (), {"loss": self.value})()

    batch = sample_calibration_tokens(SlimderConfig(calibration={"sample_count": 1, "sequence_length": 4}).calibration)[0][0]
    for value in (float("inf"), float("nan")):
        try:
            causal_lm_perplexity(BadLossModel(value), [batch])
        except ValueError as exc:
            assert "non-finite loss" in str(exc)
        else:
            raise AssertionError("causal_lm_perplexity should reject non-finite losses")
    try:
        causal_lm_perplexity(BadLossModel(1000.0), [batch])
    except ValueError as exc:
        assert "perplexity is not finite" in str(exc)
    else:
        raise AssertionError("causal_lm_perplexity should reject overflowed perplexity")

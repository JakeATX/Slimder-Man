import json
import math
from pathlib import Path

import yaml
from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM, DummyTokenizer
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


def test_run_rejects_non_dummy_transformers_without_local_preflight(monkeypatch, tmp_path: Path):
    from slimder_man import cli

    loaded = False

    def fail_if_loaded(_cfg):
        nonlocal loaded
        loaded = True
        raise AssertionError("non-dummy full run should reject before model loading")

    monkeypatch.setattr(cli, "_load_model", fail_if_loaded)
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "out")},
        teacher={"load_mode": "transformers", "model_id_or_path": "Qwen/Qwen3-Next-80B-A3B-Instruct"},
    )
    config_path = tmp_path / "qwen.yaml"
    config_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config_path), "--json"])

    assert result.exit_code != 0
    assert loaded is False


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

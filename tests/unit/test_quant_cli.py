import json
from pathlib import Path

from typer.testing import CliRunner

from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM
from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig, save_config
from slimder_man.quant.export import collect_export_hashes
from slimder_man.utils.hashing import sha256_file
from slimder_man.utils.json import read_json


def test_quantize_cli_uses_fake_backend_and_writes_manifests(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    TinyMoEForCausalLM().save_pretrained(checkpoint)
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "run")},
        quantization={"enabled": True, "target_avg_bits": 10.0},
    )
    config_path = tmp_path / "config.yaml"
    save_config(cfg, config_path)

    result = CliRunner().invoke(app, ["quantize", str(config_path), str(checkpoint), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    manifest = payload["manifest"]
    out_dir = tmp_path / "run" / "quantized"
    assert manifest["backend"] == "fake_symmetric_uniform"
    assert manifest["target_avg_bits"] == 10.0
    assert "embed_tokens.weight" in manifest["allocation"]
    assert manifest["validation"]["finite_loss"] is True
    assert manifest["artifact_hashes"]["fake_quant_manifest.json"]
    assert "quantization_manifest.json" not in manifest["artifact_hashes"]
    assert "export_artifact_hashes" not in manifest
    for name, digest in manifest["artifact_hashes"].items():
        assert digest == sha256_file(out_dir / name)
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "fake_quant_manifest.json").exists()
    assert (out_dir / "quantization_manifest.json").exists()
    export_manifest = read_json(out_dir / "quant_export_manifest.json")
    assert export_manifest["source_checkpoint"] == str(checkpoint)
    assert export_manifest["artifact_hashes"] == collect_export_hashes(out_dir)
    assert "quantization_manifest.json" in export_manifest["artifact_hashes"]
    assert export_manifest["artifact_hashes"]["quantization_manifest.json"] == sha256_file(out_dir / "quantization_manifest.json")
    reloaded = TinyMoEForCausalLM.from_pretrained(out_dir)
    assert sum(p.numel() for p in reloaded.parameters()) > 0


def test_quantize_cli_supports_hf_dummy_fake_backend(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    DummyHfMoeForCausalLM().save_pretrained(checkpoint)
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "run")},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf"},
        student={"output_format": "hf_safetensors"},
        quantization={"enabled": True, "target_avg_bits": 12.0},
    )
    config_path = tmp_path / "config.yaml"
    save_config(cfg, config_path)

    result = CliRunner().invoke(app, ["quantize", str(config_path), str(checkpoint), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    manifest = payload["manifest"]
    out_dir = tmp_path / "run" / "quantized"
    assert manifest["checkpoint_kind"] == "dummy_hf_moe"
    assert manifest["backend"] == "fake_symmetric_uniform"
    assert manifest["validation"]["finite_loss"] is True
    assert "model.layers.0.mlp.gate.weight" in manifest["allocation"]
    assert manifest["allocation"]["model.layers.0.mlp.gate.weight"] == 16
    for name, digest in manifest["artifact_hashes"].items():
        assert digest == sha256_file(out_dir / name)
    assert (out_dir / "model.safetensors").exists()
    assert (out_dir / "fake_quant_manifest.json").exists()
    export_manifest = read_json(out_dir / "quant_export_manifest.json")
    assert export_manifest["source_checkpoint"] == str(checkpoint)
    assert "quantization_manifest.json" in export_manifest["artifact_hashes"]
    reloaded = DummyHfMoeForCausalLM.from_pretrained(out_dir)
    assert sum(p.numel() for p in reloaded.parameters()) > 0

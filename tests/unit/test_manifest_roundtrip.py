from pathlib import Path

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.calibration.collectors import collect_tiny_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import compress_tiny_model
from slimder_man.compression.manifests import load_manifest
from slimder_man.config.schema import SlimderConfig


def test_manifest_roundtrip_and_reapply_shapes(tmp_path: Path):
    cfg = SlimderConfig(project={"output_dir": str(tmp_path)})
    teacher = TinyMoEForCausalLM()
    batches, _ = sample_calibration_tokens(cfg.calibration)
    cal = collect_tiny_calibration(teacher, batches)
    student, manifest = compress_tiny_model(teacher, cfg, cal, tmp_path / "ckpt")
    loaded = load_manifest(tmp_path / "ckpt" / "compression_manifest.json")
    assert loaded["schema_version"] == "1.0"
    assert "hidden_keep_indices" in loaded["width"]
    student2, _ = compress_tiny_model(teacher, cfg, cal)
    assert [tuple(p.shape) for p in student.parameters()] == [tuple(p.shape) for p in student2.parameters()]

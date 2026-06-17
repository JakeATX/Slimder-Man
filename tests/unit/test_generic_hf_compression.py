from pathlib import Path
import sys

from slimder_man.adapters.registry import get_adapter
from slimder_man.calibration.collectors import collect_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import compress_model
from slimder_man.compression.manifests import load_manifest
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.train_loop import train_causal_lm_distill

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyHfMoeForCausalLM


def test_generic_hf_dummy_compresses_saves_and_reloads(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        compression={"target": {"hidden_size": 32, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
        training={"train_steps": 1, "warmup_steps": 0},
    )
    teacher = DummyHfMoeForCausalLM()
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)

    student, manifest = compress_model(teacher, cfg, calibration, adapter=adapter, output_dir=tmp_path / "ckpt")

    assert len(student.model.layers) == 2
    assert len(student.model.layers[0].mlp.experts) == 4
    assert student.config.num_hidden_layers == 2
    assert student.config.num_experts == 4
    assert (tmp_path / "ckpt" / "model.safetensors").exists()
    loaded_manifest = load_manifest(tmp_path / "ckpt" / "compression_manifest.json")
    assert loaded_manifest["experts"]["layers"][0]["importance_metric_used"] == "soft_logits"
    assert loaded_manifest["experts"]["layers"][0]["score_vector"]
    reloaded = DummyHfMoeForCausalLM.from_pretrained(tmp_path / "ckpt")
    assert len(reloaded.model.layers) == 2
    assert manifest["param_counts"]["after"] == sum(p.numel() for p in student.parameters())

    train = train_causal_lm_distill(teacher, reloaded, cfg, tmp_path / "training", batches[:2])
    assert train["global_step"] == 1
    assert train["logs"][0]["loss"] > 0
    assert (tmp_path / "training" / "final" / "model.safetensors").exists()

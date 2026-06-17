from pathlib import Path
import sys

import pytest
import torch
from torch import nn

from slimder_man.adapters.registry import get_adapter
from slimder_man.calibration.collectors import collect_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import compress_model
from slimder_man.compression.manifests import load_manifest
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.train_loop import train_causal_lm_distill

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyHfMoeConfig, DummyHfMoeForCausalLM


def test_generic_hf_dummy_compresses_saves_and_reloads(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
        training={"train_steps": 1, "warmup_steps": 0},
    )
    teacher = DummyHfMoeForCausalLM()
    teacher.model.layers[0].self_attn.q_proj.weight.requires_grad_(False)
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)
    calibration.hidden_scores = torch.arange(teacher.config.hidden_size, dtype=torch.float32)
    depth_expert_cfg = cfg.model_copy(
        deep=True,
        update={"compression": cfg.compression.model_copy(update={"target": cfg.compression.target.model_copy(update={"hidden_size": 32})})},
    )
    depth_expert_student, _ = compress_model(teacher, depth_expert_cfg, calibration, adapter=adapter)

    student, manifest = compress_model(teacher, cfg, calibration, adapter=adapter, output_dir=tmp_path / "ckpt")

    assert len(student.model.layers) == 2
    assert len(student.model.layers[0].mlp.experts) == 4
    assert student.model.embed_tokens.weight.shape == (teacher.config.vocab_size, 24)
    assert student.lm_head.in_features == 24
    assert student.model.norm.normalized_shape == (24,)
    assert student.model.layers[0].input_layernorm.normalized_shape == (24,)
    assert student.model.layers[0].self_attn.q_proj.in_features == 24
    assert student.model.layers[0].self_attn.q_proj.out_features == 32
    assert student.model.layers[0].self_attn.q_proj.weight.requires_grad is False
    assert student.model.layers[0].self_attn.o_proj.in_features == 32
    assert student.model.layers[0].self_attn.o_proj.out_features == 24
    assert student.model.layers[0].mlp.gate.in_features == 24
    assert student.model.layers[0].mlp.gate.out_features == 4
    assert student.model.layers[0].mlp.experts[0].up_proj.in_features == 24
    assert student.model.layers[0].mlp.experts[0].down_proj.out_features == 24
    assert student.model.layers[0].mlp.shared_experts[0].up_proj.in_features == 24
    assert student.model.layers[0].mlp.shared_experts[0].down_proj.out_features == 24
    assert student.config.num_hidden_layers == 2
    assert student.config.hidden_size == 24
    assert student.config.num_experts == 4
    assert manifest["width"]["hidden_size_before"] == 32
    assert manifest["width"]["hidden_size_after"] == 24
    assert manifest["width"]["hidden_keep_indices"] == list(range(8, 32))
    assert manifest["param_counts"]["after"] < sum(p.numel() for p in depth_expert_student.parameters())
    assert (tmp_path / "ckpt" / "model.safetensors").exists()
    loaded_manifest = load_manifest(tmp_path / "ckpt" / "compression_manifest.json")
    assert loaded_manifest["experts"]["layers"][0]["importance_metric_used"] == "soft_logits"
    assert loaded_manifest["experts"]["layers"][0]["score_vector"]
    assert loaded_manifest["width"]["hidden_keep_indices"] == list(range(8, 32))
    reloaded = DummyHfMoeForCausalLM.from_pretrained(tmp_path / "ckpt")
    assert len(reloaded.model.layers) == 2
    assert reloaded.config.hidden_size == 24
    assert reloaded.config.attention_hidden_size == 32
    assert reloaded.model.layers[0].self_attn.q_proj.in_features == 24
    assert reloaded.model.layers[0].mlp.experts[0].down_proj.out_features == 24
    assert manifest["param_counts"]["after"] == sum(p.numel() for p in student.parameters())
    reloaded_out = reloaded(input_ids=batches[0][:1], labels=batches[0][:1])
    assert reloaded_out.logits.shape == (1, cfg.calibration.sequence_length, teacher.config.vocab_size)
    assert reloaded_out.loss is not None and torch.isfinite(reloaded_out.loss)

    train = train_causal_lm_distill(teacher, reloaded, cfg, tmp_path / "training", batches[:2])
    assert train["global_step"] == 1
    assert train["logs"][0]["loss"] > 0
    assert (tmp_path / "training" / "final" / "model.safetensors").exists()


def test_generic_hidden_slicing_preserves_tied_embeddings(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM(DummyHfMoeConfig(tie_word_embeddings=True))
    assert teacher.lm_head.weight is teacher.model.embed_tokens.weight
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 0, "routed_experts": 6, "routed_top_k": 2}},
    )
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)
    calibration.hidden_scores = torch.arange(teacher.config.hidden_size, dtype=torch.float32)

    student, _ = compress_model(teacher, cfg, calibration, adapter=adapter, output_dir=tmp_path / "tied")

    assert student.lm_head.weight is student.model.embed_tokens.weight
    assert student.lm_head.weight.shape == (teacher.config.vocab_size, 24)
    out = student(input_ids=batches[0][:1], labels=batches[0][:1])
    assert out.loss is not None and torch.isfinite(out.loss)
    reloaded = DummyHfMoeForCausalLM.from_pretrained(tmp_path / "tied")
    assert reloaded.lm_head.weight is reloaded.model.embed_tokens.weight
    reloaded_out = reloaded(input_ids=batches[0][:1], labels=batches[0][:1])
    assert reloaded_out.loss is not None and torch.isfinite(reloaded_out.loss)


def test_generic_hidden_slicing_rejects_ambiguous_hidden_linear(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    teacher.ambiguous_hidden_linear = nn.Linear(teacher.config.hidden_size, teacher.config.hidden_size, bias=False)
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 0, "routed_experts": 6, "routed_top_k": 2}},
    )
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)

    with pytest.raises(ValueError, match="Unsupported hidden-size Linear modules"):
        compress_model(teacher, cfg, calibration, adapter=adapter)


@pytest.mark.parametrize("out_features", [32, 64])
def test_generic_hidden_slicing_rejects_nonstructural_gate_proj(tmp_path: Path, out_features: int):
    teacher = DummyHfMoeForCausalLM()
    teacher.gate_proj = nn.Linear(teacher.config.hidden_size, out_features, bias=False)
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 0, "routed_experts": 6, "routed_top_k": 2}},
    )
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)

    with pytest.raises(ValueError, match="Unsupported hidden-size Linear modules"):
        compress_model(teacher, cfg, calibration, adapter=adapter)


@pytest.mark.parametrize("module_name", ["down_proj", "w2"])
def test_generic_hidden_slicing_rejects_nonstructural_down_projection(tmp_path: Path, module_name: str):
    teacher = DummyHfMoeForCausalLM()
    setattr(teacher, module_name, nn.Linear(64, teacher.config.hidden_size, bias=False))
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 0, "routed_experts": 6, "routed_top_k": 2}},
    )
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)

    with pytest.raises(ValueError, match="Unsupported hidden-size Linear modules"):
        compress_model(teacher, cfg, calibration, adapter=adapter)

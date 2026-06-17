import sys
from pathlib import Path

from slimder_man.calibration.collectors import collect_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import compress_model
from slimder_man.config.schema import SlimderConfig
from slimder_man.adapters.qwen3_next import Qwen3NextAdapter

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyHfMoeConfig, DummyHfMoeForCausalLM


def test_qwen3_next_adapter_uses_structural_moe_detection():
    config = DummyHfMoeConfig(model_type="qwen3_next")
    model = DummyHfMoeForCausalLM(config)
    adapter = Qwen3NextAdapter()

    assert adapter.match(model)
    layers = adapter.iter_moe_layers(model)
    assert len(layers) == config.num_hidden_layers
    assert len(adapter.get_routed_experts(layers[0])) == config.num_experts
    assert len(adapter.get_shared_experts(layers[0])) == config.num_shared_experts
    assert adapter.get_router(layers[0]).out_features == config.num_experts

    info = adapter.describe_architecture(model)
    assert len(info.moe_layers) == config.num_hidden_layers
    assert info.moe_layers[0].num_routed_experts == config.num_experts
    assert info.moe_layers[0].num_shared_experts == config.num_shared_experts
    assert info.moe_layers[0].top_k == config.num_experts_per_tok


def test_qwen3_next_fixture_compresses_width_depth_and_experts(tmp_path: Path):
    config = DummyHfMoeConfig(model_type="qwen3_next")
    model = DummyHfMoeForCausalLM(config)
    adapter = Qwen3NextAdapter()
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-qwen3-next"},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=model.config.vocab_size)
    calibration = collect_calibration(model, batches, adapter)

    student, manifest = compress_model(model, cfg, calibration, adapter=adapter, output_dir=tmp_path / "qwen")

    assert len(student.model.layers) == 2
    assert student.config.hidden_size == 24
    assert student.config.num_experts == 4
    assert student.model.embed_tokens.weight.shape == (config.vocab_size, 24)
    assert student.model.layers[0].input_layernorm.normalized_shape == (24,)
    assert student.model.layers[0].self_attn.q_proj.in_features == 24
    assert student.model.layers[0].self_attn.q_proj.out_features == 32
    assert student.model.layers[0].self_attn.o_proj.in_features == 32
    assert student.model.layers[0].self_attn.o_proj.out_features == 24
    assert student.model.layers[0].mlp.gate.in_features == 24
    assert student.model.layers[0].mlp.experts[0].up_proj.in_features == 24
    assert student.model.layers[0].mlp.experts[0].down_proj.out_features == 24
    assert len(student.model.layers[0].mlp.experts) == 4
    assert manifest["width"]["hidden_size_after"] == 24
    reloaded = DummyHfMoeForCausalLM.from_pretrained(tmp_path / "qwen")
    assert reloaded.config.attention_hidden_size == 32
    out = reloaded(input_ids=batches[0][:1])
    assert out.logits.shape[-1] == config.vocab_size

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from slimder_man.calibration.collectors import collect_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import compress_model
from slimder_man.config.schema import SlimderConfig
from slimder_man.adapters.qwen3_next import Qwen3NextAdapter

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyHfMoeConfig, DummyHfMoeForCausalLM


class DenseMlp(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(x)))


def _official_tiny_qwen3_next():
    transformers = pytest.importorskip("transformers")
    Qwen3NextConfig = getattr(transformers, "Qwen3NextConfig", None)
    Qwen3NextForCausalLM = getattr(transformers, "Qwen3NextForCausalLM", None)
    if Qwen3NextConfig is None or Qwen3NextForCausalLM is None:
        pytest.skip("Installed transformers does not expose Qwen3Next official classes")
    config = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts_per_tok=2,
        num_experts=4,
        layer_types=["linear_attention", "full_attention"],
        use_cache=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    torch.manual_seed(2026)
    return Qwen3NextForCausalLM(config)


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


def test_official_qwen3_next_packed_experts_router_and_width_slices():
    model = _official_tiny_qwen3_next()
    adapter = Qwen3NextAdapter()
    info = adapter.describe_architecture(model)

    assert info.model_type == "qwen3_next"
    assert info.block_kinds == ["linear_attention", "full_attention"]
    assert len(info.moe_layers) == 2
    assert info.moe_layers[0].num_routed_experts == 4
    assert info.moe_layers[0].top_k == 2
    moe = adapter.iter_moe_layers(model)[0]
    assert adapter.get_router(moe).weight.shape == (4, 16)
    assert len(adapter.get_routed_experts(moe)) == 4
    original_router = moe.gate.weight.detach().clone()
    original_gate_up = moe.experts.gate_up_proj.detach().clone()
    original_down = moe.experts.down_proj.detach().clone()
    keep = torch.arange(12)

    adapter.slice_hidden_channels(model, keep)

    assert model.config.hidden_size == 12
    assert model.model.layers[0].linear_attn.hidden_size == 12
    assert model.model.layers[0].linear_attn.in_proj_qkvz.weight.shape == (32, 12)
    assert model.model.layers[0].linear_attn.in_proj_ba.weight.shape == (4, 12)
    assert model.model.layers[0].linear_attn.out_proj.weight.shape == (12, 8)
    assert model.model.layers[1].self_attn.q_proj.weight.shape == (32, 12)
    assert model.model.layers[1].self_attn.o_proj.weight.shape == (12, 16)
    assert torch.equal(moe.gate.weight.detach(), original_router.index_select(1, keep))
    assert moe.gate.hidden_dim == 12
    assert torch.equal(moe.experts.gate_up_proj.detach(), original_gate_up.index_select(2, keep))
    assert torch.equal(moe.experts.down_proj.detach(), original_down.index_select(1, keep))
    out = model(input_ids=torch.randint(0, 128, (1, 8)), labels=torch.randint(0, 128, (1, 8)))
    assert torch.isfinite(out.logits).all()
    assert out.logits.shape == (1, 8, 128)


def test_qwen3_next_adapter_reports_actual_sparse_moe_block_indices():
    config = DummyHfMoeConfig(model_type="qwen3_next", num_hidden_layers=3)
    model = DummyHfMoeForCausalLM(config)
    model.model.layers[1].mlp = DenseMlp(config.hidden_size, config.intermediate_size)
    adapter = Qwen3NextAdapter()

    layers = adapter.iter_moe_layers(model)
    info = adapter.describe_architecture(model)

    assert len(layers) == 2
    assert [layer.layer_idx for layer in info.moe_layers] == [0, 2]
    assert info.num_layers == 3


def test_qwen3_next_sparse_moe_compression_uses_stable_layer_indices(tmp_path: Path):
    config = DummyHfMoeConfig(model_type="qwen3_next", num_hidden_layers=3)
    model = DummyHfMoeForCausalLM(config)
    model.model.layers[1].mlp = DenseMlp(config.hidden_size, config.intermediate_size)
    adapter = Qwen3NextAdapter()
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-sparse-qwen3-next"},
        compression={"target": {"hidden_size": 32, "remove_last_n_layers": 0, "routed_experts": 4, "routed_top_k": 2}},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=model.config.vocab_size)
    calibration = collect_calibration(model, batches, adapter)

    student, manifest = compress_model(model, cfg, calibration, adapter=adapter)

    assert calibration.expert_layer_indices == [0, 2]
    assert [layer["layer_idx"] for layer in manifest["experts"]["layers"]] == [0, 2]
    assert len(adapter.iter_moe_layers(student)) == 2


def test_qwen3_next_adapter_prefers_config_layer_types_and_shared_intermediate_field():
    config = DummyHfMoeConfig(model_type="qwen3_next", num_hidden_layers=4, num_shared_experts=1)
    model = DummyHfMoeForCausalLM(config)
    model.config.num_shared_experts = None
    model.config.shared_expert_intermediate_size = 512
    model.config.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    adapter = Qwen3NextAdapter()

    info = adapter.describe_architecture(model)

    assert info.block_kinds == ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    assert info.num_linear_attention_layers == 3
    assert info.num_full_attention_layers == 1
    assert info.moe_layers[0].num_shared_experts == 1


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


def test_official_qwen3_next_calibrates_compresses_saves_and_reloads(tmp_path: Path):
    model = _official_tiny_qwen3_next()
    adapter = Qwen3NextAdapter()
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "official-tiny-qwen3-next"},
        student={"output_format": "hf_safetensors"},
        calibration={"sample_count": 2, "sequence_length": 8},
        compression={"target": {"hidden_size": 12, "remove_last_n_layers": 0, "routed_experts": 2, "routed_top_k": 1}},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=model.config.vocab_size)
    calibration = collect_calibration(model, batches, adapter)

    student, manifest = compress_model(model, cfg, calibration, adapter=adapter, output_dir=tmp_path / "official_qwen")

    assert student.config.hidden_size == 12
    assert student.config.num_hidden_layers == 2
    assert student.config.num_experts == 2
    assert student.config.num_experts_per_tok == 1
    assert student.model.layers[0].mlp.gate.weight.shape == (2, 12)
    assert student.model.layers[0].mlp.gate.hidden_dim == 12
    assert student.model.layers[0].mlp.experts.gate_up_proj.shape == (2, 16, 12)
    assert student.model.layers[0].mlp.experts.down_proj.shape == (2, 12, 8)
    assert manifest["target"] == {"hidden_size": 12, "remove_last_n_layers": 0, "routed_experts": 2, "top_k": 1}
    assert (tmp_path / "official_qwen" / "model.safetensors").exists()

    transformers = pytest.importorskip("transformers")
    reloaded = transformers.Qwen3NextForCausalLM.from_pretrained(tmp_path / "official_qwen")
    out = reloaded(input_ids=batches[0][:1], labels=batches[0][:1])
    assert reloaded.config.layer_types == ["linear_attention", "full_attention"]
    assert out.logits.shape == (1, 8, 128)
    assert torch.isfinite(out.logits).all()
    assert out.loss is not None and torch.isfinite(out.loss)


def test_qwen3_next_adapter_destructive_methods_are_explicitly_covered(tmp_path: Path):
    config = DummyHfMoeConfig(model_type="qwen3_next")
    model = DummyHfMoeForCausalLM(config)
    adapter = Qwen3NextAdapter()

    adapter.drop_blocks(model, [0, 2])
    assert len(model.model.layers) == 2
    assert model.model.layers[0] is not model.model.layers[1]

    moe = adapter.iter_moe_layers(model)[0]
    original_experts = adapter.get_routed_experts(moe)
    router_rows = adapter.get_router(moe).weight.detach().clone()[:3]
    adapter.replace_experts(moe, original_experts[:3], router_rows, new_top_k=2)
    assert len(adapter.get_routed_experts(moe)) == 3
    assert adapter.get_router(moe).weight.shape == (3, config.hidden_size)
    assert torch.equal(adapter.get_router(moe).weight.detach(), router_rows.to(adapter.get_router(moe).weight.dtype))
    assert moe.num_experts_per_tok == 2

    manifest = {
        "target": {"hidden_size": 24, "routed_experts": 3, "top_k": 2},
        "depth": {"kept_block_indices": [0, 2]},
    }
    adapter.update_config_after_compression(model, manifest)
    assert model.config.hidden_size == 24
    assert model.config.num_hidden_layers == 2
    assert model.config.num_experts == 3
    assert model.config.num_experts_per_tok == 2

    adapter.save_pretrained(model, str(tmp_path / "qwen_save"))
    assert (tmp_path / "qwen_save" / "model.safetensors").exists()
    assert (tmp_path / "qwen_save" / "config.json").exists()

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
from slimder_man.config.schema import SlimderConfig, save_config
from slimder_man.distill.train_loop import train_causal_lm_distill
from slimder_man.utils.hashing import sha256_file

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyHfMoeConfig, DummyHfMoeForCausalLM, DummyTokenizer


class MultiFileTokenizer(DummyTokenizer):
    def __init__(self, marker: str = "first"):
        self.marker = marker

    def save_pretrained(self, output_dir: str | Path):
        path = Path(output_dir)
        super().save_pretrained(path)
        files = []
        for name, content in {
            "tokenizer.json": f'{{"marker": "{self.marker}"}}',
            "special_tokens_map.json": '{"pad_token": "<pad>"}',
            "vocab.txt": "<pad>\nhello\nworld\n",
        }.items():
            target = path / name
            target.write_text(content, encoding="utf-8")
            files.append(str(target))
        return files


class ConfigClobberingTokenizer(DummyTokenizer):
    def save_pretrained(self, output_dir: str | Path):
        super().save_pretrained(output_dir)
        (Path(output_dir) / "config.json").write_text('{"not": "a model config"}', encoding="utf-8")


class ConfigDeletingTokenizer(DummyTokenizer):
    def save_pretrained(self, output_dir: str | Path):
        super().save_pretrained(output_dir)
        (Path(output_dir) / "config.json").unlink()


class NotATokenizer:
    pass


def test_generic_hf_dummy_compresses_saves_and_reloads(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        student={"output_format": "hf_safetensors"},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
        training={"train_steps": 1, "warmup_steps": 0},
    )
    teacher = DummyHfMoeForCausalLM()
    teacher.model.layers[0].self_attn.q_proj.weight.requires_grad_(False)
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)
    calibration.hidden_scores = torch.arange(teacher.config.hidden_size, dtype=torch.float32)
    analysis_dir = tmp_path / "analysis"
    from slimder_man.calibration.artifacts import write_calibration_artifacts

    write_calibration_artifacts(
        analysis_dir,
        cfg,
        calibration,
        {"type": "synthetic", "sample_hashes": ["fixture"], "dataset_hash": "0" * 64},
        {"model_type": "dummy_hf_moe"},
    )
    depth_expert_cfg = cfg.model_copy(
        deep=True,
        update={"compression": cfg.compression.model_copy(update={"target": cfg.compression.target.model_copy(update={"hidden_size": 32})})},
    )
    depth_expert_student, _ = compress_model(teacher, depth_expert_cfg, calibration, adapter=adapter)

    student, manifest = compress_model(
        teacher,
        cfg,
        calibration,
        adapter=adapter,
        output_dir=tmp_path / "ckpt",
        tokenizer=DummyTokenizer(),
        calibration_manifest_path=analysis_dir / "calibration_manifest.json",
    )

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
    assert manifest["provenance"]["normalized_config_sha256"]
    assert manifest["provenance"]["package_version"]
    assert manifest["progressive"]["stages"] == cfg.progressive.stages
    assert manifest["calibration_artifacts"]["manifest_sha256"] == sha256_file(analysis_dir / "calibration_manifest.json")
    assert manifest["calibration_artifacts"]["calibration"]["sample_hashes"] == ["fixture"]
    first_layer = manifest["experts"]["layers"][0]
    assert Path(first_layer["score_artifact"]["path"]).exists()
    assert first_layer["score_artifact"]["tensor"] == "soft_logits"
    assert Path(first_layer["similarity_artifact"]["path"]).exists()
    assert first_layer["similarity_artifact"]["metric"] == "router_weights"
    assert (tmp_path / "ckpt" / "model.safetensors").exists()
    assert (tmp_path / "ckpt" / "tokenizer_config.json").exists()
    loaded_manifest = load_manifest(tmp_path / "ckpt" / "compression_manifest.json")
    assert loaded_manifest["student_output_format"] == "hf_safetensors"
    assert loaded_manifest["calibration_artifacts"]["manifest_sha256"] == sha256_file(analysis_dir / "calibration_manifest.json")
    assert loaded_manifest["experts"]["layers"][0]["importance_metric_used"] == "soft_logits"
    assert loaded_manifest["experts"]["layers"][0]["score_vector"]
    assert loaded_manifest["width"]["hidden_keep_indices"] == list(range(8, 32))
    assert loaded_manifest["tokenizer"]["saved"] is True
    assert loaded_manifest["tokenizer"]["artifact_hashes"]["tokenizer_config.json"] == sha256_file(tmp_path / "ckpt" / "tokenizer_config.json")
    assert loaded_manifest["artifact_hashes"]["tokenizer_config.json"] == sha256_file(tmp_path / "ckpt" / "tokenizer_config.json")
    for name, digest in loaded_manifest["artifact_hashes"].items():
        assert digest == sha256_file(tmp_path / "ckpt" / name)
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


def test_generic_hf_compression_honors_torch_output_format(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        student={"output_format": "torch"},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
        calibration={"sample_count": 2, "sequence_length": 8},
    )
    teacher = DummyHfMoeForCausalLM()
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)
    out_dir = tmp_path / "torch_ckpt"

    _, manifest = compress_model(teacher, cfg, calibration, adapter=adapter, output_dir=out_dir, tokenizer=DummyTokenizer())

    assert manifest["student_output_format"] == "torch"
    assert (out_dir / "pytorch_model.bin").exists()
    assert not (out_dir / "model.safetensors").exists()
    loaded_manifest = load_manifest(out_dir / "compression_manifest.json")
    assert loaded_manifest["student_output_format"] == "torch"
    assert loaded_manifest["artifact_hashes"]["pytorch_model.bin"] == sha256_file(out_dir / "pytorch_model.bin")
    reloaded = DummyHfMoeForCausalLM.from_pretrained(out_dir)
    out = reloaded(input_ids=batches[0], labels=batches[0])
    assert out.loss is not None and torch.isfinite(out.loss)


def test_generic_hf_compression_honors_depth_remove_fraction(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        compression={"target": {"hidden_size": 32, "depth_remove_fraction": 0.5, "remove_last_n_layers": 0, "routed_experts": 6, "routed_top_k": 2}},
        calibration={"sample_count": 2, "sequence_length": 8},
    )
    teacher = DummyHfMoeForCausalLM(DummyHfMoeConfig(num_hidden_layers=4))
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)

    student, manifest = compress_model(teacher, cfg, calibration, adapter=adapter)

    assert len(student.model.layers) == 2
    assert manifest["target"]["remove_last_n_layers"] == 2
    assert manifest["depth"]["kept_block_indices"] == [0, 1]


def test_cli_transformers_compress_saves_tokenizer_artifacts(monkeypatch, tmp_path: Path):
    from typer.testing import CliRunner

    from slimder_man import cli

    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "run"), "paper_faithful": True},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        student={"output_format": "hf_safetensors"},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
        calibration={"sample_count": 2, "sequence_length": 8},
    )
    config_path = tmp_path / "hf_dummy.yaml"
    save_config(cfg, config_path)
    monkeypatch.setattr(cli, "_load_model", lambda _cfg: DummyHfMoeForCausalLM())
    monkeypatch.setattr(cli, "_load_tokenizer", lambda _cfg: DummyTokenizer())

    result = CliRunner().invoke(cli.app, ["compress", "--config", str(config_path), "--json"])

    assert result.exit_code == 0, result.output
    ckpt = tmp_path / "run" / "checkpoints" / "stage_1_compressed"
    manifest = load_manifest(ckpt / "compression_manifest.json")
    assert (ckpt / "model.safetensors").exists()
    assert (ckpt / "config.json").exists()
    assert (ckpt / "tokenizer_config.json").exists()
    assert manifest["tokenizer"]["saved"] is True
    assert manifest["tokenizer"]["artifact_hashes"]["tokenizer_config.json"] == sha256_file(ckpt / "tokenizer_config.json")
    assert manifest["artifact_hashes"]["tokenizer_config.json"] == sha256_file(ckpt / "tokenizer_config.json")


def test_checked_in_hf_dummy_config_runs_without_monkeypatch(tmp_path: Path):
    from typer.testing import CliRunner

    from slimder_man import cli

    config_path = Path("src/slimder_man/config/examples/hf_dummy.yaml").resolve()
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli.app, ["compress", "--config", str(config_path), "--json"])
        ckpt = Path("runs/hf_dummy_moe_smoke/checkpoints/stage_1_compressed")
        assert result.exit_code == 0, result.output
        assert (ckpt / "model.safetensors").exists()
        assert (ckpt / "config.json").exists()
        assert (ckpt / "tokenizer_config.json").exists()
        manifest = load_manifest(ckpt / "compression_manifest.json")
        assert manifest["teacher_model"] == "dummy-hf-moe"
        assert manifest["tokenizer"]["saved"] is True
        assert manifest["artifact_hashes"]["tokenizer_config.json"] == sha256_file(ckpt / "tokenizer_config.json")


def test_hf_compression_hashes_multifile_tokenizer_and_reruns(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        student={"output_format": "hf_safetensors"},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
        calibration={"sample_count": 2, "sequence_length": 8},
    )
    teacher = DummyHfMoeForCausalLM()
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)
    out_dir = tmp_path / "ckpt"

    compress_model(teacher, cfg, calibration, adapter=adapter, output_dir=out_dir, tokenizer=MultiFileTokenizer("first"))
    _, manifest = compress_model(teacher, cfg, calibration, adapter=adapter, output_dir=out_dir, tokenizer=MultiFileTokenizer("second"))

    expected_tokenizer_files = {"tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.txt"}
    assert expected_tokenizer_files.issubset(manifest["tokenizer"]["artifact_hashes"])
    assert expected_tokenizer_files.issubset(manifest["artifact_hashes"])
    assert (out_dir / "tokenizer.json").read_text(encoding="utf-8") == '{"marker": "second"}'
    for name in expected_tokenizer_files | {"config.json", "model.safetensors"}:
        assert manifest["artifact_hashes"][name] == sha256_file(out_dir / name)
    reloaded = DummyHfMoeForCausalLM.from_pretrained(out_dir)
    tokenizer_ids = MultiFileTokenizer()(["hello world"], max_length=4)["input_ids"]
    out = reloaded(input_ids=tokenizer_ids, labels=tokenizer_ids)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_hf_compression_rejects_invalid_or_clobbering_tokenizer(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path), "paper_faithful": True},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        compression={"target": {"hidden_size": 24, "remove_last_n_layers": 1, "routed_experts": 4, "routed_top_k": 2}},
        calibration={"sample_count": 2, "sequence_length": 8},
    )
    teacher = DummyHfMoeForCausalLM()
    adapter = get_adapter(teacher)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    calibration = collect_calibration(teacher, batches, adapter)

    with pytest.raises(ValueError, match="does not expose save_pretrained"):
        compress_model(teacher, cfg, calibration, adapter=adapter, output_dir=tmp_path / "bad_tokenizer", tokenizer=NotATokenizer())
    with pytest.raises(ValueError, match="modified model config.json"):
        compress_model(teacher, cfg, calibration, adapter=adapter, output_dir=tmp_path / "clobber", tokenizer=ConfigClobberingTokenizer())
    with pytest.raises(ValueError, match="removed model config.json"):
        compress_model(teacher, cfg, calibration, adapter=adapter, output_dir=tmp_path / "delete", tokenizer=ConfigDeletingTokenizer())


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

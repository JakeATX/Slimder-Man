from pathlib import Path
import sys

import torch

from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.config.schema import CalibrationConfig, DatasetConfig

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyTokenizer


def test_text_calibration_sampling_can_use_tokenizer(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text("alpha beta gamma delta", encoding="utf-8")
    cfg = CalibrationConfig(
        dataset=DatasetConfig(type="text", path=str(path)),
        sample_count=2,
        sequence_length=3,
    )
    tokenizer = DummyTokenizer()

    batches, manifest = sample_calibration_tokens(cfg, vocab_size=tokenizer.vocab_size, tokenizer=tokenizer)

    assert len(batches) == 2
    assert all(batch.shape == (1, 3) for batch in batches)
    assert torch.equal(batches[0], torch.tensor([[4, 155, 1]]))
    assert manifest["sample_count"] == 2
    assert manifest["sequence_length"] == 3
    assert len(manifest["dataset_hash"]) == 64
    assert manifest["tokenizer_fingerprint"]["class"] == "DummyTokenizer"
    assert len(manifest["tokenizer_fingerprint"]["fingerprint_sha256"]) == 64


def test_tokenizer_fingerprint_distinguishes_same_class_different_behavior(tmp_path: Path):
    class OffsetTokenizer(DummyTokenizer):
        def __init__(self, offset: int):
            super().__init__()
            self.offset = offset

        def encode(self, text: str, add_special_tokens: bool = False):
            return [token + self.offset for token in super().encode(text, add_special_tokens=add_special_tokens)]

    path = tmp_path / "sample.txt"
    path.write_text("alpha beta gamma delta", encoding="utf-8")
    cfg = CalibrationConfig(dataset=DatasetConfig(type="text", path=str(path)), sample_count=1, sequence_length=4)

    _, manifest_a = sample_calibration_tokens(cfg, vocab_size=128, tokenizer=OffsetTokenizer(0))
    _, manifest_b = sample_calibration_tokens(cfg, vocab_size=128, tokenizer=OffsetTokenizer(7))

    assert manifest_a["tokenizer"] == manifest_b["tokenizer"] == "OffsetTokenizer"
    assert manifest_a["tokenizer_fingerprint"]["fingerprint_sha256"] != manifest_b["tokenizer_fingerprint"]["fingerprint_sha256"]


def test_tokenized_pt_sampling_uses_weights_only_load(monkeypatch, tmp_path: Path):
    path = tmp_path / "tokens.pt"
    path.write_bytes(b"placeholder")
    seen = {}

    def fake_load(load_path, **kwargs):
        seen.update(kwargs)
        assert Path(load_path) == path
        return torch.arange(12)

    monkeypatch.setattr(torch, "load", fake_load)
    cfg = CalibrationConfig(
        dataset=DatasetConfig(type="tokenized", path=str(path)),
        sample_count=2,
        sequence_length=4,
    )

    batches, manifest = sample_calibration_tokens(cfg, vocab_size=128)

    assert seen["map_location"] == "cpu"
    assert seen["weights_only"] is True
    assert [batch.shape for batch in batches] == [(1, 4), (1, 4)]
    assert manifest["source"]["path"] == str(path)

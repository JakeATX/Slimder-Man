from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from slimder_man.config.schema import CalibrationConfig


def synthetic_token_batches(sample_count: int, sequence_length: int, vocab_size: int = 128, seed: int = 1337) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab_size, (1, sequence_length), generator=g) for _ in range(sample_count)]


def _encode_text(text: str, vocab_size: int, tokenizer: object | None = None) -> list[int]:
    if tokenizer is None:
        return [ord(ch) % vocab_size for ch in text]
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        encoded = tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) % vocab_size for token in ids]


def sample_calibration_tokens(cfg: CalibrationConfig, vocab_size: int = 128, tokenizer: object | None = None) -> tuple[list[torch.Tensor], dict]:
    if cfg.dataset.type == "synthetic":
        batches = synthetic_token_batches(cfg.sample_count, cfg.sequence_length, vocab_size=vocab_size, seed=cfg.seed)
    elif cfg.dataset.type == "text":
        text = Path(cfg.dataset.path or "").read_text(encoding="utf-8")
        ids = _encode_text(text, vocab_size, tokenizer)
        batches = []
        for i in range(cfg.sample_count):
            start = (i * cfg.sequence_length) % max(1, len(ids))
            chunk = (ids[start : start + cfg.sequence_length] + [0] * cfg.sequence_length)[: cfg.sequence_length]
            batches.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    elif cfg.dataset.type == "jsonl":
        rows = [json.loads(line).get(cfg.dataset.text_field, "") for line in Path(cfg.dataset.path or "").read_text(encoding="utf-8").splitlines()]
        batches = []
        for i, row in enumerate(rows[: cfg.sample_count]):
            ids = _encode_text(row, vocab_size, tokenizer)
            chunk = (ids + [0] * cfg.sequence_length)[: cfg.sequence_length]
            batches.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    elif cfg.dataset.type == "hf_dataset":
        from datasets import load_dataset

        ds = load_dataset(cfg.dataset.name or "", split=cfg.dataset.split)
        batches = []
        for row in ds.select(range(min(cfg.sample_count, len(ds)))):
            text = str(row.get(cfg.dataset.text_field, ""))
            ids = _encode_text(text, vocab_size, tokenizer)
            chunk = (ids + [0] * cfg.sequence_length)[: cfg.sequence_length]
            batches.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    elif cfg.dataset.type == "parquet":
        import pandas as pd

        df = pd.read_parquet(cfg.dataset.path or "")
        batches = []
        for text in df[cfg.dataset.text_field].astype(str).head(cfg.sample_count):
            ids = _encode_text(text, vocab_size, tokenizer)
            chunk = (ids + [0] * cfg.sequence_length)[: cfg.sequence_length]
            batches.append(torch.tensor(chunk, dtype=torch.long).unsqueeze(0))
    elif cfg.dataset.type == "tokenized":
        path = Path(cfg.dataset.path or "")
        if path.suffix in {".pt", ".pth"}:
            data = torch.load(path, map_location="cpu")
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
        tensor = torch.as_tensor(data, dtype=torch.long).reshape(-1)
        batches = []
        for i in range(cfg.sample_count):
            start = (i * cfg.sequence_length) % max(1, tensor.numel())
            chunk = tensor[start : start + cfg.sequence_length]
            if chunk.numel() < cfg.sequence_length:
                chunk = torch.cat([chunk, torch.zeros(cfg.sequence_length - chunk.numel(), dtype=torch.long)])
            batches.append(chunk.unsqueeze(0) % vocab_size)
    else:
        raise ValueError(f"Unsupported dataset type {cfg.dataset.type}")
    digest = hashlib.sha256()
    for batch in batches:
        digest.update(batch.numpy().tobytes())
    return batches, {"sample_count": len(batches), "sequence_length": cfg.sequence_length, "dataset_hash": digest.hexdigest()}

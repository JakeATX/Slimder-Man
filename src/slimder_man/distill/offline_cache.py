from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch


def full_logits_cache_key(input_ids: torch.Tensor) -> str:
    ids = input_ids.detach().cpu().to(torch.int64).contiguous()
    shape = ",".join(str(x) for x in ids.shape).encode("utf-8")
    return hashlib.sha256(shape + b"\0" + ids.numpy().tobytes()).hexdigest()


class OfflineFullLogitsCache:
    def __init__(self, entries: dict[str, torch.Tensor]) -> None:
        self.entries = {key: value.detach().cpu().to(torch.float32).contiguous() for key, value in entries.items()}

    @classmethod
    def from_path(cls, path: str | Path) -> "OfflineFullLogitsCache":
        cache_path = Path(path)
        if not cache_path.exists():
            raise ValueError(f"offline full-logit cache not found: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return cls(_entries_from_cache_payload(data))

    def teacher_output(self, input_ids: torch.Tensor) -> SimpleNamespace:
        key = full_logits_cache_key(input_ids)
        if key not in self.entries:
            raise ValueError(f"offline full-logit cache missing exact entry for input_ids sha256={key}")
        logits = self.entries[key]
        expected_prefix = tuple(input_ids.shape)
        if tuple(logits.shape[:2]) != expected_prefix:
            raise ValueError(f"offline full-logit cache shape mismatch: logits={tuple(logits.shape)}, input_ids={expected_prefix}")
        return SimpleNamespace(logits=logits.to(device=input_ids.device))


class OfflineTopKLogitsCache:
    def __init__(self, entries: dict[str, tuple[torch.Tensor, torch.Tensor]], vocab_size: int, fill_value: float) -> None:
        self.entries = {
            key: (
                indices.detach().cpu().to(torch.long).contiguous(),
                values.detach().cpu().to(torch.float32).contiguous(),
            )
            for key, (indices, values) in entries.items()
        }
        self.vocab_size = int(vocab_size)
        self.fill_value = float(fill_value)

    @classmethod
    def from_path(cls, path: str | Path) -> "OfflineTopKLogitsCache":
        cache_path = Path(path)
        if not cache_path.exists():
            raise ValueError(f"offline top-k logit cache not found: {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        entries, vocab_size, fill_value = _topk_entries_from_cache_payload(data)
        return cls(entries, vocab_size, fill_value)

    def teacher_output(self, input_ids: torch.Tensor) -> SimpleNamespace:
        key = full_logits_cache_key(input_ids)
        if key not in self.entries:
            raise ValueError(f"offline top-k logit cache missing exact entry for input_ids sha256={key}")
        indices, values = self.entries[key]
        if tuple(indices.shape[:2]) != tuple(input_ids.shape) or tuple(values.shape) != tuple(indices.shape):
            raise ValueError(
                f"offline top-k logit cache shape mismatch: indices={tuple(indices.shape)}, "
                f"values={tuple(values.shape)}, input_ids={tuple(input_ids.shape)}"
            )
        logits = torch.full((*input_ids.shape, self.vocab_size), self.fill_value, dtype=torch.float32)
        logits.scatter_(-1, indices, values)
        return SimpleNamespace(logits=logits.to(device=input_ids.device))


def write_full_logits_cache(path: str | Path, pairs: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    entries = []
    keys = []
    for input_ids, logits in pairs:
        key = full_logits_cache_key(input_ids)
        keys.append(key)
        entries.append(
            {
                "key": key,
                "input_ids": input_ids.detach().cpu().to(torch.int64),
                "logits": logits.detach().cpu().to(torch.float32),
            }
        )
    payload = {"format": "slimder_full_logits_cache_v1", "entries": entries}
    torch.save(payload, path)
    return {"path": str(path), "format": payload["format"], "num_entries": len(entries), "keys": keys}


def write_topk_logits_cache(
    path: str | Path,
    pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
    top_k: int,
    fill_value: float = -1.0e4,
) -> dict:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    entries = []
    keys = []
    vocab_size: int | None = None
    for input_ids, logits in pairs:
        if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(input_ids.shape):
            raise ValueError("top-k cache logits must have shape [batch, seq, vocab] matching input_ids")
        if top_k > logits.shape[-1]:
            raise ValueError("top_k must not exceed logits vocabulary dimension")
        vocab_size = int(logits.shape[-1]) if vocab_size is None else vocab_size
        if vocab_size != int(logits.shape[-1]):
            raise ValueError("all top-k cache entries must use the same vocab size")
        values, indices = torch.topk(logits.detach().cpu().to(torch.float32), k=top_k, dim=-1)
        key = full_logits_cache_key(input_ids)
        keys.append(key)
        entries.append(
            {
                "key": key,
                "input_ids": input_ids.detach().cpu().to(torch.int64),
                "topk_indices": indices.to(torch.int64),
                "topk_values": values,
            }
        )
    if vocab_size is None:
        raise ValueError("offline top-k logit cache contains no entries")
    payload = {
        "format": "slimder_topk_logits_cache_v1",
        "vocab_size": vocab_size,
        "top_k": int(top_k),
        "fill_value": float(fill_value),
        "entries": entries,
    }
    torch.save(payload, path)
    return {"path": str(path), "format": payload["format"], "num_entries": len(entries), "keys": keys, "top_k": int(top_k), "vocab_size": vocab_size}


def _entries_from_cache_payload(data) -> dict[str, torch.Tensor]:
    if not isinstance(data, dict):
        raise ValueError("offline full-logit cache must be a dict payload")
    if data.get("format") != "slimder_full_logits_cache_v1":
        raise ValueError("offline full-logit cache format must be slimder_full_logits_cache_v1")
    entries: dict[str, torch.Tensor] = {}
    for idx, entry in enumerate(data.get("entries", [])):
        if not isinstance(entry, dict) or "input_ids" not in entry or "logits" not in entry:
            raise ValueError(f"offline full-logit cache entry {idx} must contain input_ids and logits")
        input_ids = torch.as_tensor(entry["input_ids"], dtype=torch.long)
        logits = torch.as_tensor(entry["logits"], dtype=torch.float32)
        key = str(entry.get("key") or full_logits_cache_key(input_ids))
        expected = full_logits_cache_key(input_ids)
        if key != expected:
            raise ValueError(f"offline full-logit cache entry {idx} key does not match input_ids")
        if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(input_ids.shape):
            raise ValueError(f"offline full-logit cache entry {idx} has incompatible logits/input_ids shapes")
        entries[key] = logits
    if not entries:
        raise ValueError("offline full-logit cache contains no entries")
    return entries


def _topk_entries_from_cache_payload(data) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], int, float]:
    if not isinstance(data, dict):
        raise ValueError("offline top-k logit cache must be a dict payload")
    if data.get("format") != "slimder_topk_logits_cache_v1":
        raise ValueError("offline top-k logit cache format must be slimder_topk_logits_cache_v1")
    vocab_size = int(data.get("vocab_size") or 0)
    if vocab_size <= 0:
        raise ValueError("offline top-k logit cache must declare a positive vocab_size")
    fill_value = float(data.get("fill_value", -1.0e4))
    if not torch.isfinite(torch.tensor(fill_value)):
        raise ValueError("offline top-k logit cache fill_value must be finite")
    entries: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for idx, entry in enumerate(data.get("entries", [])):
        if not isinstance(entry, dict) or "input_ids" not in entry or "topk_indices" not in entry or "topk_values" not in entry:
            raise ValueError(f"offline top-k logit cache entry {idx} must contain input_ids, topk_indices, and topk_values")
        input_ids = torch.as_tensor(entry["input_ids"], dtype=torch.long)
        indices = torch.as_tensor(entry["topk_indices"], dtype=torch.long)
        values = torch.as_tensor(entry["topk_values"], dtype=torch.float32)
        key = str(entry.get("key") or full_logits_cache_key(input_ids))
        expected = full_logits_cache_key(input_ids)
        if key != expected:
            raise ValueError(f"offline top-k logit cache entry {idx} key does not match input_ids")
        if indices.ndim != 3 or values.shape != indices.shape or tuple(indices.shape[:2]) != tuple(input_ids.shape):
            raise ValueError(f"offline top-k logit cache entry {idx} has incompatible top-k/input_ids shapes")
        if indices.shape[-1] <= 0:
            raise ValueError(f"offline top-k logit cache entry {idx} must contain at least one top-k value")
        if indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= vocab_size):
            raise ValueError(f"offline top-k logit cache entry {idx} has indices outside vocab_size")
        if not torch.isfinite(values).all():
            raise ValueError(f"offline top-k logit cache entry {idx} contains non-finite values")
        if key in entries:
            raise ValueError(f"offline top-k logit cache contains duplicate entry for input_ids sha256={key}")
        entries[key] = (indices, values)
    if not entries:
        raise ValueError("offline top-k logit cache contains no entries")
    return entries, vocab_size, fill_value

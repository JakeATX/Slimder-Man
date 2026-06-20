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

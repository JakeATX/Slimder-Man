from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch


@dataclass
class TokenizedCalibration:
    batches: list[torch.Tensor]
    manifest: dict


def tokenize_text_samples(texts: list[str], tokenizer, sample_count: int, sequence_length: int, seed: int = 1337, packing: bool = True) -> TokenizedCalibration:
    selected = list(texts)
    g = torch.Generator().manual_seed(seed)
    if len(selected) > sample_count:
        perm = torch.randperm(len(selected), generator=g).tolist()[:sample_count]
        selected = [selected[i] for i in perm]
    encoded = tokenizer(selected, max_length=sequence_length, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]
    batches = [encoded[i : i + 1].clone() for i in range(encoded.shape[0])]
    digest = hashlib.sha256()
    sample_hashes = []
    for text, batch in zip(selected, batches):
        sample_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        sample_hashes.append(sample_hash)
        digest.update(sample_hash.encode("ascii"))
        digest.update(batch.numpy().tobytes())
    return TokenizedCalibration(
        batches=batches,
        manifest={
            "sample_count": len(batches),
            "sequence_length": sequence_length,
            "packing": packing,
            "seed": seed,
            "sample_hashes": sample_hashes,
            "dataset_hash": digest.hexdigest(),
            "tokenizer": tokenizer.__class__.__name__,
        },
    )

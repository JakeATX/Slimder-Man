from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def redact_secret(text: str) -> str:
    markers = ["hf_", "sk-", "ghp_", "github_pat_"]
    out = text
    for marker in markers:
        idx = out.find(marker)
        while idx != -1:
            end = idx
            while end < len(out) and (out[end].isalnum() or out[end] in "_-"):
                end += 1
            out = out[:idx] + marker + "***REDACTED***" + out[end:]
            idx = out.find(marker, idx + len(marker) + 14)
    return out

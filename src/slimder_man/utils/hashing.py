from __future__ import annotations

import hashlib
import re
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def redact_secret(text: str) -> str:
    out = text
    patterns = [
        ("hf_", r"hf_[A-Za-z0-9]{8,}"),
        ("sk-", r"sk-[A-Za-z0-9_-]{8,}"),
        ("ghp_", r"ghp_[A-Za-z0-9]{8,}"),
        ("github_pat_", r"github_pat_[A-Za-z0-9_]{8,}"),
    ]
    for marker, pattern in patterns:
        out = re.sub(pattern, marker + "***REDACTED***", out)
    return out

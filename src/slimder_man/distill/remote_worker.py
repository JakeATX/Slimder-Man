from __future__ import annotations

import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from urllib import request

import numpy as np
import torch

from slimder_man.config.schema import RuntimeWorkerConfig


@dataclass
class RemoteWorkerLogitsClient:
    api_url: str
    auth_token: str | None = None
    timeout_seconds: float = 60.0

    @classmethod
    def from_config(cls, cfg: RuntimeWorkerConfig) -> "RemoteWorkerLogitsClient":
        if not cfg.api_url:
            raise ValueError("runtime.worker.api_url is required for kd.teacher_mode=remote_worker_full_logits")
        token = cfg.auth_token
        if token is None and cfg.auth_token_env:
            token = os.environ.get(cfg.auth_token_env)
        return cls(api_url=cfg.api_url, auth_token=token, timeout_seconds=cfg.timeout_seconds)

    def teacher_output(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.fetch_logits(input_ids))

    def fetch_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        payload = json.dumps(
            {
                "input_ids": input_ids.detach().cpu().to(torch.long).tolist(),
                "response_format": "binary_float32",
            }
        ).encode("utf-8")
        req = request.Request(
            self.api_url.rstrip("/") + "/v1/teacher_logits",
            data=payload,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}),
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = response.read()
            shape_header = response.headers.get("X-Slimder-Logits-Shape")
            fmt = response.headers.get("X-Slimder-Logits-Format")
        if fmt != "float32_le" or not shape_header:
            raise ValueError("remote worker returned unsupported logits transport")
        try:
            shape = tuple(int(part) for part in shape_header.split(",") if part)
        except ValueError as exc:
            raise ValueError(f"remote worker returned invalid logits shape header: {shape_header}") from exc
        expected = int(np.prod(shape, dtype=np.int64)) if shape else 0
        actual = len(body) // np.dtype("<f4").itemsize
        if expected <= 0 or expected != actual:
            raise ValueError(f"remote worker logits payload shape mismatch: shape={shape_header}, values={actual}")
        logits = np.frombuffer(body, dtype="<f4").reshape(shape)
        return torch.from_numpy(logits.copy()).to(device=input_ids.device)

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import redact_secret


@dataclass
class WorkerAPIRunResult:
    backend: str
    status: str
    api_url: str
    job: dict[str, Any] | None = None
    request_payload: dict[str, Any] | None = None
    dry_run: bool = False


class WorkerAPIClient:
    def __init__(self, api_url: str, auth_token: str | None = None, timeout_seconds: float = 60.0) -> None:
        self.api_url = api_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, cfg) -> "WorkerAPIClient":
        if not cfg.api_url:
            raise ValueError("runtime.worker.api_url is required for backend=worker")
        token = cfg.auth_token
        if token is None and cfg.auth_token_env:
            token = os.environ.get(cfg.auth_token_env)
        return cls(cfg.api_url, auth_token=token, timeout_seconds=cfg.timeout_seconds)

    def create_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._json("POST", "/v1/jobs", payload)

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{job_id}", None)

    def logs(self, job_id: str) -> dict[str, Any]:
        return self._json("GET", f"/v1/jobs/{job_id}/logs", None)

    def stop(self, job_id: str) -> dict[str, Any]:
        return self._json("POST", f"/v1/jobs/{job_id}/stop", {})

    def _json(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.api_url + path,
            data=body,
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}),
            },
            method=method,
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))


class WorkerAPIRunner:
    def __init__(
        self,
        config_path: str | Path,
        cfg: SlimderConfig,
        client: WorkerAPIClient | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.cfg = cfg
        self.client = client or WorkerAPIClient.from_config(cfg.runtime.worker)

    def payload(self) -> dict[str, Any]:
        return {
            "command": "run",
            "config_text": self.config_path.read_text(encoding="utf-8"),
            "config_filename": self.config_path.name,
            "args": ["--json"],
            "artifact_paths": [],
        }

    def launch(self, dry_run: bool = False) -> WorkerAPIRunResult:
        payload = self.payload()
        safe_url = redact_secret(self.client.api_url)
        safe_payload = _redacted_payload(_summarize_payload(payload))
        if dry_run:
            return WorkerAPIRunResult("worker", "dry_run", safe_url, request_payload=safe_payload, dry_run=True)
        job = self.client.create_job(payload)
        return WorkerAPIRunResult("worker", str(job.get("status", "submitted")), safe_url, job=_redacted_payload(job))

    def status(self, job_id: str) -> dict[str, Any]:
        return _redacted_payload(self.client.get_job(job_id))

    def logs(self, job_id: str) -> dict[str, Any]:
        return _redacted_payload(self.client.logs(job_id))

    def stop(self, job_id: str) -> dict[str, Any]:
        return _redacted_payload(self.client.stop(job_id))


def _redacted_payload(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secret(value)
    if isinstance(value, list):
        return [_redacted_payload(item) for item in value]
    if isinstance(value, dict):
        return {key: _redacted_payload(item) for key, item in value.items()}
    return value


def _summarize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    safe = dict(payload)
    config_text = safe.pop("config_text", None)
    if config_text is not None:
        safe["config_text_bytes"] = len(config_text.encode("utf-8"))
    return safe

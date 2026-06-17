import sys
import time
from pathlib import Path

import numpy as np
import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig, save_config
from slimder_man.orchestration.worker_api import create_worker_app


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_worker_job_api_executes_local_subprocess_and_persists_state(tmp_path: Path):
    job_root = tmp_path / "worker"
    artifact_dir = tmp_path / "artifacts"
    client = TestClient(create_worker_app(job_root=job_root))

    created = client.post(
        "/v1/jobs",
        json={
            "command": sys.executable,
            "args": ["-c", "from pathlib import Path; print('worker-ok'); Path(r'%s').mkdir()" % artifact_dir],
            "artifact_paths": [str(artifact_dir)],
        },
    ).json()

    assert created["status"] in {"queued", "running", "succeeded"}
    assert created["request"]["command"] == sys.executable
    assert str(artifact_dir.resolve()) in created["artifact_paths"]

    finished = _wait_for_terminal(client, created["id"])
    assert finished["status"] == "succeeded"
    assert finished["returncode"] == 0
    assert Path(finished["log_path"]).exists()
    assert str(artifact_dir.resolve()) in finished["artifact_paths"]

    logs = client.get(f"/v1/jobs/{created['id']}/logs").json()
    assert "worker-ok" in logs["logs"]

    restarted_client = TestClient(create_worker_app(job_root=job_root))
    persisted = restarted_client.get(f"/v1/jobs/{created['id']}").json()
    assert persisted["status"] == "succeeded"
    assert persisted["log_path"] == finished["log_path"]


def test_worker_job_api_can_cancel_running_local_subprocess(tmp_path: Path):
    client = TestClient(create_worker_app(job_root=tmp_path / "worker"))
    created = client.post(
        "/v1/jobs",
        json={"command": sys.executable, "args": ["-u", "-c", "import time; print('started'); time.sleep(30)"]},
    ).json()

    stopped = client.post(f"/v1/jobs/{created['id']}/stop").json()
    assert stopped["status"] == "cancelled"

    fetched = client.get(f"/v1/jobs/{created['id']}").json()
    assert fetched["status"] == "cancelled"
    assert Path(fetched["log_path"]).exists()


def test_worker_run_command_uses_slimder_cli_with_positional_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "out")},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy/full-model-dry-run"},
    )
    save_config(cfg, config_path)
    client = TestClient(create_worker_app(job_root=tmp_path / "worker"))

    created = client.post(
        "/v1/jobs",
        json={"command": "run", "config_path": str(config_path), "args": ["--dry-run", "--json"]},
    ).json()

    finished = _wait_for_terminal(client, created["id"])
    assert finished["status"] == "succeeded"
    logs = client.get(f"/v1/jobs/{created['id']}/logs").json()["logs"]
    assert any('"status": "dry_run"' in line for line in logs)
    assert str((tmp_path / "out").resolve()) in finished["artifact_paths"]


def test_worker_auth_guards_v1_endpoints_and_accepts_bearer_token(tmp_path: Path):
    client = TestClient(create_worker_app(job_root=tmp_path / "worker", auth_token="secret-token"))

    assert client.get("/healthz").status_code == 200
    assert client.post("/v1/preflight").status_code == 401
    assert client.post("/v1/preflight", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.post("/v1/preflight", headers={"Authorization": "Bearer secret-token"})
    assert ok.status_code == 200


class _Teacher:
    def eval(self):
        return self

    def __call__(self, input_ids):
        batch, seq = input_ids.shape
        logits = torch.arange(batch * seq * 5, dtype=torch.float32).reshape(batch, seq, 5)
        return type("Out", (), {"logits": logits})()


def test_worker_teacher_logits_binary_transport_and_json_fallback(tmp_path: Path):
    client = TestClient(create_worker_app(job_root=tmp_path / "worker", auth_token="token", teacher_model=_Teacher()))
    headers = {"X-Slimder-Worker-Token": "token"}

    binary = client.post("/v1/teacher_logits", json={"input_ids": [[1, 2, 3]]}, headers=headers)

    assert binary.status_code == 200
    assert binary.headers["content-type"] == "application/octet-stream"
    assert binary.headers["x-slimder-logits-format"] == "float32_le"
    assert binary.headers["x-slimder-logits-shape"] == "1,3,5"
    decoded = np.frombuffer(binary.content, dtype="<f4").reshape(1, 3, 5)
    assert decoded[0, 2, 4] == 14.0

    fallback = client.post(
        "/v1/teacher_logits",
        json={"input_ids": [[1]], "response_format": "json_nested"},
        headers=headers,
    )
    assert fallback.status_code == 200
    assert fallback.json()["format"] == "nested_float32"
    assert fallback.json()["shape"] == [1, 1, 5]


def test_worker_cli_json_reports_auth_requirement():
    result = CliRunner().invoke(app, ["worker", "--auth-token", "secret-token", "--json"])

    assert result.exit_code == 0, result.output
    assert '"auth_required": true' in result.output


def test_worker_cli_json_reports_env_auth_requirement(monkeypatch):
    monkeypatch.setenv("SLIMDER_WORKER_TOKEN", "env-token")

    result = CliRunner().invoke(app, ["worker", "--json"])

    assert result.exit_code == 0, result.output
    assert '"auth_required": true' in result.output

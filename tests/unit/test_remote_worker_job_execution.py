import json
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from slimder_man.cli import app
from slimder_man.config.schema import SlimderConfig, save_config
from slimder_man.distill.remote_worker import RemoteWorkerLogitsClient
from slimder_man.orchestration.worker_client import WorkerAPIClient, WorkerAPIRunner
from slimder_man.orchestration.worker_api import create_worker_app


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 5.0, headers: dict | None = None) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/v1/jobs/{job_id}", headers=headers).json()
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


def test_worker_job_preserves_cuda_visible_devices(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    client = TestClient(create_worker_app(job_root=tmp_path / "worker"))

    created = client.post(
        "/v1/jobs",
        json={
            "command": sys.executable,
            "args": ["-c", "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES'))"],
        },
    ).json()

    finished = _wait_for_terminal(client, created["id"])
    logs = client.get(f"/v1/jobs/{created['id']}/logs").json()["logs"]

    assert finished["runtime"]["cuda_visible_devices"] == "0,1"
    assert "0,1" in logs


def test_worker_preflight_reports_visible_devices(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    client = TestClient(create_worker_app(job_root=tmp_path / "worker"))

    preflight = client.post("/v1/preflight").json()

    assert preflight["cuda_visible_devices"] == "2"


def test_worker_job_api_lists_and_zips_artifacts(tmp_path: Path):
    job_root = tmp_path / "worker"
    artifact_dir = tmp_path / "artifacts"
    nested = artifact_dir / "nested"
    nested.mkdir(parents=True)
    (nested / "result.txt").write_text("artifact-ok", encoding="utf-8")
    try:
        (nested / "linked.txt").symlink_to(nested / "result.txt")
    except OSError:
        pass
    client = TestClient(create_worker_app(job_root=job_root, auth_token="token"))
    headers = {"Authorization": "Bearer token"}

    created = client.post(
        "/v1/jobs",
        json={
            "command": sys.executable,
            "args": ["-c", "print('done')"],
            "artifact_paths": [str(artifact_dir)],
        },
        headers=headers,
    ).json()
    finished = _wait_for_terminal(client, created["id"], headers=headers)

    listing = client.get(f"/v1/jobs/{finished['id']}/artifacts", headers=headers).json()
    zipped = client.get(f"/v1/jobs/{finished['id']}/artifacts.zip", headers=headers)

    assert listing["artifacts"][0]["files"] == [{"path": "nested/result.txt", "bytes": 11}]
    assert zipped.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(BytesIO(zipped.content)) as zf:
        assert "artifacts/nested/linked.txt" not in zf.namelist()
        assert zf.read("artifacts/nested/result.txt") == b"artifact-ok"


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


def test_worker_job_api_materializes_config_text_for_remote_launch(tmp_path: Path):
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path / "remote out")},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy/full-model-dry-run"},
    )
    client = TestClient(create_worker_app(job_root=tmp_path / "worker"))

    created = client.post(
        "/v1/jobs",
        json={
            "command": "run",
            "config_text": yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False),
            "config_filename": "remote.yaml",
            "args": ["--dry-run", "--json"],
        },
    ).json()

    finished = _wait_for_terminal(client, created["id"])
    assert finished["status"] == "succeeded"
    request = finished["request"]
    assert request["config_text"] is None
    assert request["config_path"].endswith("remote.yaml")
    assert Path(request["config_path"]).exists()
    assert str((tmp_path / "remote out").resolve()) in finished["artifact_paths"]
    logs = client.get(f"/v1/jobs/{created['id']}/logs").json()["logs"]
    assert any('"status": "dry_run"' in line for line in logs)


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


def test_remote_worker_logits_client_decodes_binary_transport(monkeypatch):
    captured = {}
    logits = np.arange(6, dtype="<f4").reshape(1, 2, 3)

    class FakeResponse:
        headers = {"X-Slimder-Logits-Format": "float32_le", "X-Slimder-Logits-Shape": "1,2,3"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return logits.tobytes()

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        return FakeResponse()

    monkeypatch.setattr("slimder_man.distill.remote_worker.request.urlopen", fake_urlopen)

    client = RemoteWorkerLogitsClient("http://worker", auth_token="secret-token", timeout_seconds=7)
    out = client.fetch_logits(torch.tensor([[1, 2]]))

    assert captured["url"] == "http://worker/v1/teacher_logits"
    assert captured["timeout"] == 7
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert b"binary_float32" in captured["body"]
    assert out.shape == (1, 2, 3)
    assert out[0, 1, 2].item() == 5.0


def test_remote_worker_logits_client_rejects_shape_mismatch(monkeypatch):
    class FakeResponse:
        headers = {"X-Slimder-Logits-Format": "float32_le", "X-Slimder-Logits-Shape": "1,2,99"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return np.arange(6, dtype="<f4").tobytes()

    monkeypatch.setattr("slimder_man.distill.remote_worker.request.urlopen", lambda req, timeout: FakeResponse())

    client = RemoteWorkerLogitsClient("http://worker")
    try:
        client.fetch_logits(torch.tensor([[1, 2]]))
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("client should reject malformed binary logits payloads")


def test_worker_api_runner_submits_config_text_and_redacts_token(tmp_path: Path):
    config_path = tmp_path / "worker.yaml"
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "out")},
        runtime={"worker": {"api_url": "http://worker.example", "auth_token": "hf_secret123"}},
    )
    save_config(cfg, config_path)
    captured = {}

    class FakeClient(WorkerAPIClient):
        def __init__(self):
            super().__init__("http://worker.example", auth_token="hf_secret123")

        def create_job(self, payload):
            captured["payload"] = payload
            return {"id": "job-1", "status": "running", "token": "hf_secret123"}

    result = WorkerAPIRunner(config_path, cfg, client=FakeClient()).launch()

    assert result.status == "running"
    assert result.job["token"] == "hf_***REDACTED***"
    assert captured["payload"]["command"] == "run"
    assert captured["payload"]["config_filename"] == "worker.yaml"
    assert "paper_faithful: false" in captured["payload"]["config_text"]
    assert captured["payload"]["args"] == ["--json"]


def test_worker_api_runner_dry_run_summarizes_config_text(tmp_path: Path):
    config_path = tmp_path / "worker.yaml"
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "out")},
        runtime={"worker": {"api_url": "http://worker.example", "auth_token": "plain-secret"}},
    )
    save_config(cfg, config_path)

    result = WorkerAPIRunner(config_path, cfg, client=WorkerAPIClient("http://worker.example")).launch(dry_run=True)

    assert result.dry_run is True
    assert result.request_payload is not None
    assert "config_text" not in result.request_payload
    assert result.request_payload["config_text_bytes"] > 0


def test_worker_api_client_preflight_uses_auth_header(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"python": true}'

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = req.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("slimder_man.orchestration.worker_client.request.urlopen", fake_urlopen)

    result = WorkerAPIClient("http://worker", auth_token="secret-token", timeout_seconds=3).preflight()

    assert result == {"python": True}
    assert captured["url"] == "http://worker/v1/preflight"
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["body"] == b"{}"
    assert captured["timeout"] == 3


def test_worker_api_runner_syncs_artifact_zip_and_rejects_unsafe_paths(tmp_path: Path):
    config_path = tmp_path / "worker.yaml"
    cfg = SlimderConfig(project={"paper_faithful": False}, runtime={"worker": {"api_url": "http://worker.example"}})
    save_config(cfg, config_path)

    class FakeClient(WorkerAPIClient):
        def __init__(self, body: bytes):
            super().__init__("http://worker.example")
            self.body = body

        def artifacts_zip(self, job_id):
            assert job_id == "job-1"
            return self.body

    good_buffer = BytesIO()
    with zipfile.ZipFile(good_buffer, "w") as zf:
        zf.writestr("run/result.txt", "ok")

    sync = WorkerAPIRunner(config_path, cfg, client=FakeClient(good_buffer.getvalue())).sync_outputs("job-1", tmp_path / "synced")

    assert Path(sync["files"][0]).read_text(encoding="utf-8") == "ok"

    bad_buffer = BytesIO()
    with zipfile.ZipFile(bad_buffer, "w") as zf:
        zf.writestr("../escape.txt", "bad")

    try:
        WorkerAPIRunner(config_path, cfg, client=FakeClient(bad_buffer.getvalue())).sync_outputs("job-1", tmp_path / "bad")
    except ValueError as exc:
        assert "unsafe path" in str(exc)
    else:
        raise AssertionError("worker sync should reject unsafe zip paths")


def test_cli_launch_worker_uses_worker_api_runner(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "worker.yaml"
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "out")},
        runtime={"worker": {"api_url": "http://worker.example"}},
    )
    save_config(cfg, config_path)
    captured = {}

    class FakeRunner:
        def __init__(self, config, cfg):
            captured["config"] = config
            captured["cfg"] = cfg

        def launch(self, dry_run=False):
            captured["dry_run"] = dry_run
            return type(
                "Run",
                (),
                {
                    "backend": "worker",
                    "status": "running",
                    "api_url": "http://worker.example",
                    "job": {"id": "job-1", "status": "running"},
                    "dry_run": dry_run,
                },
            )()

    monkeypatch.setattr("slimder_man.cli.WorkerAPIRunner", FakeRunner)

    result = CliRunner().invoke(app, ["launch", str(config_path), "--backend", "worker", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["backend"] == "worker"
    assert payload["job"]["id"] == "job-1"
    assert captured["config"] == config_path
    assert captured["cfg"].runtime.worker.api_url == "http://worker.example"
    assert captured["dry_run"] is False


def test_cli_worker_lifecycle_commands_use_worker_api_runner(monkeypatch, tmp_path: Path):
    config_path = tmp_path / "worker.yaml"
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path / "out")},
        runtime={"worker": {"api_url": "http://worker.example"}},
    )
    save_config(cfg, config_path)
    calls = []

    class FakeRunner:
        def __init__(self, config, cfg):
            calls.append(("init", config, cfg.runtime.worker.api_url))

        def preflight(self):
            calls.append(("preflight",))
            return {"python": True}

        def status(self, job_id):
            calls.append(("status", job_id))
            return {"id": job_id, "status": "running"}

        def logs(self, job_id):
            calls.append(("logs", job_id))
            return {"id": job_id, "logs": ["ok"]}

        def artifacts(self, job_id):
            calls.append(("artifacts", job_id))
            return {"id": job_id, "artifacts": []}

        def sync_outputs(self, job_id, out):
            calls.append(("sync", job_id, out))
            return {"job_id": job_id, "output_dir": str(out), "files": []}

        def stop(self, job_id):
            calls.append(("stop", job_id))
            return {"id": job_id, "status": "cancelled"}

    monkeypatch.setattr("slimder_man.cli.WorkerAPIRunner", FakeRunner)
    runner = CliRunner()

    preflight = runner.invoke(app, ["worker-preflight", "--config", str(config_path), "--json"])
    status = runner.invoke(app, ["worker-status", "--config", str(config_path), "--job-id", "job-1", "--json"])
    logs = runner.invoke(app, ["worker-logs", "--config", str(config_path), "--job-id", "job-1", "--json"])
    artifacts = runner.invoke(app, ["worker-artifacts", "--config", str(config_path), "--job-id", "job-1", "--json"])
    sync = runner.invoke(app, ["worker-sync", "--config", str(config_path), "--job-id", "job-1", "--out", str(tmp_path / "synced"), "--json"])
    stop = runner.invoke(app, ["worker-stop", "--config", str(config_path), "--job-id", "job-1", "--json"])

    assert preflight.exit_code == 0, preflight.output
    assert status.exit_code == 0, status.output
    assert logs.exit_code == 0, logs.output
    assert artifacts.exit_code == 0, artifacts.output
    assert sync.exit_code == 0, sync.output
    assert stop.exit_code == 0, stop.output
    assert json.loads(preflight.output)["preflight"] == {"python": True}
    assert json.loads(status.output)["job"]["status"] == "running"
    assert json.loads(logs.output)["logs"]["logs"] == ["ok"]
    assert json.loads(artifacts.output)["artifacts"]["artifacts"] == []
    assert json.loads(sync.output)["sync"]["files"] == []
    assert json.loads(stop.output)["job"]["status"] == "cancelled"
    assert ("status", "job-1") in calls
    assert ("logs", "job-1") in calls
    assert ("artifacts", "job-1") in calls
    assert ("sync", "job-1", tmp_path / "synced") in calls
    assert ("stop", "job-1") in calls


def test_worker_cli_json_reports_auth_requirement():
    result = CliRunner().invoke(app, ["worker", "--auth-token", "secret-token", "--json"])

    assert result.exit_code == 0, result.output
    assert '"host": "127.0.0.1"' in result.output
    assert '"auth_required": true' in result.output


def test_worker_cli_json_reports_env_auth_requirement(monkeypatch):
    monkeypatch.setenv("SLIMDER_WORKER_TOKEN", "env-token")

    result = CliRunner().invoke(app, ["worker", "--json"])

    assert result.exit_code == 0, result.output
    assert '"auth_required": true' in result.output


def test_worker_cli_rejects_public_bind_without_auth(monkeypatch):
    monkeypatch.delenv("SLIMDER_WORKER_TOKEN", raising=False)

    result = CliRunner().invoke(app, ["worker", "--host", "0.0.0.0", "--json"])

    assert result.exit_code != 0
    assert "refuses non-local bind without auth" in result.output


def test_worker_cli_allows_public_bind_with_auth():
    result = CliRunner().invoke(app, ["worker", "--host", "0.0.0.0", "--auth-token", "secret-token", "--json"])

    assert result.exit_code == 0, result.output
    assert '"host": "0.0.0.0"' in result.output
    assert '"auth_required": true' in result.output

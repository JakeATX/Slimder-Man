import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

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

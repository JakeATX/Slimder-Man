from fastapi.testclient import TestClient

from slimder_man.orchestration.worker_api import create_worker_app


def test_worker_job_api_records_requests_but_does_not_execute_jobs():
    client = TestClient(create_worker_app())
    created = client.post("/v1/jobs", json={"config_path": "config.yaml", "command": "run"}).json()
    assert created["status"] == "queued"
    assert created["request"] == {"config_path": "config.yaml", "command": "run"}

    fetched = client.get(f"/v1/jobs/{created['id']}").json()
    assert fetched == created
    assert client.get(f"/v1/jobs/{created['id']}/logs").json()["logs"] == []

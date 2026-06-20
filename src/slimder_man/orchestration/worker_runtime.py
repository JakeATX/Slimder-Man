from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from slimder_man.config.schema import load_config


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class WorkerJobRequest(BaseModel):
    config_path: str | None = None
    config_text: str | None = None
    config_filename: str = "launch_config.yaml"
    command: str = "run"
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    artifact_paths: list[str] = Field(default_factory=list)


@dataclass
class WorkerJobRuntime:
    process: subprocess.Popen[str]
    thread: threading.Thread


class WorkerJobStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or ".slimder_worker").resolve()
        self.jobs_dir = self.root / "jobs"
        self.logs_dir = self.root / "logs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._running: dict[str, WorkerJobRuntime] = {}

    def create(self, req: WorkerJobRequest) -> dict[str, Any]:
        job_id = uuid.uuid4().hex
        req = self._materialize_config_text(job_id, req)
        now = time.time()
        job = {
            "id": job_id,
            "status": "queued",
            "request": req.model_dump(),
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "log_path": str(self._log_path(job_id)),
            "artifact_paths": self._artifact_paths(req),
            "error": None,
        }
        self._write(job)
        return job

    def _materialize_config_text(self, job_id: str, req: WorkerJobRequest) -> WorkerJobRequest:
        if req.config_text is None:
            return req
        config_dir = self.jobs_dir / job_id
        config_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(req.config_filename).name or "launch_config.yaml"
        config_path = config_dir / safe_name
        config_path.write_text(req.config_text, encoding="utf-8")
        return req.model_copy(update={"config_path": str(config_path), "config_text": None})

    def start(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] == "missing":
            return job
        if job["status"] != "queued":
            return job

        req = WorkerJobRequest.model_validate(job["request"])
        argv = self._argv(req)
        cwd = str(Path(req.cwd).resolve()) if req.cwd else None
        env = os.environ.copy()
        visible_devices = env.get("CUDA_VISIBLE_DEVICES", "<unset>")

        log_file = self._log_path(job_id).open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            log_file.write(f"failed to start job: {exc}\n")
            log_file.close()
            return self._update(job_id, status="failed", finished_at=time.time(), error=str(exc), returncode=None)

        job = self._update(
            job_id,
            status="running",
            started_at=time.time(),
            pid=process.pid,
            runtime={"cuda_visible_devices": visible_devices},
        )

        def wait_for_process() -> None:
            try:
                returncode = process.wait()
            finally:
                log_file.close()
            with self._lock:
                current = self.get(job_id)
                if current["status"] == "cancelled":
                    self._running.pop(job_id, None)
                    return
                status = "succeeded" if returncode == 0 else "failed"
                self._update(job_id, status=status, finished_at=time.time(), returncode=returncode)
                self._running.pop(job_id, None)

        thread = threading.Thread(target=wait_for_process, name=f"slimder-worker-job-{job_id}", daemon=True)
        with self._lock:
            self._running[job_id] = WorkerJobRuntime(process=process, thread=thread)
        thread.start()
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        path = self._job_path(job_id)
        with self._lock:
            if not path.exists():
                return {"id": job_id, "status": "missing"}
            return json.loads(path.read_text(encoding="utf-8"))

    def logs(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        log_path = Path(job.get("log_path") or self._log_path(job_id))
        if not log_path.exists():
            return {"id": job_id, "logs": [], "log_path": str(log_path)}
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"id": job_id, "logs": lines, "log_path": str(log_path)}

    def stop(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            runtime = self._running.get(job_id)
            job = self.get(job_id)
            if job["status"] == "missing":
                return job
            if job["status"] in TERMINAL_STATUSES:
                return job
            if runtime is not None and runtime.process.poll() is None:
                runtime.process.terminate()
                try:
                    runtime.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    runtime.process.kill()
                    runtime.process.wait(timeout=3)
            self._running.pop(job_id, None)
            return self._update(job_id, status="cancelled", finished_at=time.time(), returncode=job.get("returncode"))

    def _argv(self, req: WorkerJobRequest) -> list[str]:
        if req.command == "run":
            if req.config_path is None:
                raise ValueError("config_path is required when command='run'")
            return [sys.executable, "-m", "slimder_man.cli", "run", req.config_path, *req.args]
        return [req.command, *req.args]

    def _artifact_paths(self, req: WorkerJobRequest) -> list[str]:
        paths = [str(Path(path).resolve()) for path in req.artifact_paths]
        if req.config_path:
            try:
                cfg = load_config(req.config_path)
            except Exception:
                return paths
            output_dir = Path(cfg.project.output_dir)
            if not output_dir.is_absolute() and req.cwd:
                output_dir = Path(req.cwd) / output_dir
            paths.append(str(output_dir.resolve()))
        return paths

    def _update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        job = self.get(job_id)
        if job["status"] == "missing":
            return job
        job.update(changes)
        job["updated_at"] = time.time()
        if "artifact_paths" not in changes:
            req = WorkerJobRequest.model_validate(job["request"])
            job["artifact_paths"] = self._artifact_paths(req)
        self._write(job)
        return job

    def _write(self, job: dict[str, Any]) -> None:
        with self._lock:
            path = self._job_path(job["id"])
            tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
            tmp_path.write_text(json.dumps(job, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp_path, path)

    def _job_path(self, job_id: str) -> Path:
        return self.jobs_dir / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self.logs_dir / f"{job_id}.log"

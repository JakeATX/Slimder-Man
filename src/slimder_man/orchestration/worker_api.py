from __future__ import annotations

import hmac
import io
import os
from pathlib import Path
from typing import Literal
import zipfile

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

from slimder_man.orchestration.worker_runtime import WorkerJobRequest, WorkerJobStore


class JobRequest(WorkerJobRequest):
    pass


class TensorRequest(BaseModel):
    input_ids: list[list[int]]
    response_format: Literal["binary_float32", "json_nested"] = "binary_float32"


def create_worker_app(
    model_id_or_path: str | None = None,
    job_root: str | Path | None = None,
    auth_token: str | None = None,
    teacher_model: object | None = None,
) -> FastAPI:
    app = FastAPI(title="Slimder Man Worker")
    jobs = WorkerJobStore(job_root)
    token = auth_token if auth_token is not None else os.environ.get("SLIMDER_WORKER_TOKEN")
    teacher = teacher_model
    if teacher is None and model_id_or_path:
        from transformers import AutoModelForCausalLM

        teacher = AutoModelForCausalLM.from_pretrained(model_id_or_path, trust_remote_code=True)
        teacher.eval()

    @app.middleware("http")
    async def require_worker_token(request: Request, call_next):
        if token and request.url.path.startswith("/v1/"):
            provided = _request_token(request)
            if not provided or not hmac.compare_digest(provided, token):
                return Response("unauthorized", status_code=401)
        return await call_next(request)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.post("/v1/preflight")
    def preflight():
        import shutil
        import torch

        return {"python": True, "cuda_available": torch.cuda.is_available(), "nvidia_smi": shutil.which("nvidia-smi") is not None}

    @app.post("/v1/jobs")
    def create_job(req: JobRequest):
        job = jobs.create(req)
        return jobs.start(job["id"])

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        return jobs.get(job_id)

    @app.get("/v1/jobs/{job_id}/logs")
    def logs(job_id: str):
        return jobs.logs(job_id)

    @app.get("/v1/jobs/{job_id}/artifacts")
    def artifacts(job_id: str):
        job = jobs.get(job_id)
        if job["status"] == "missing":
            raise HTTPException(status_code=404, detail="job not found")
        return {"id": job_id, "artifacts": _artifact_listing(job)}

    @app.get("/v1/jobs/{job_id}/artifacts.zip")
    def artifacts_zip(job_id: str):
        job = jobs.get(job_id)
        if job["status"] == "missing":
            raise HTTPException(status_code=404, detail="job not found")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root_text in job.get("artifact_paths", []):
                root = Path(root_text)
                if root.is_file():
                    if not root.is_symlink():
                        zf.write(root, arcname=root.name)
                elif root.is_dir():
                    for path in sorted(root.rglob("*")):
                        if path.is_file() and not path.is_symlink():
                            zf.write(path, arcname=str(Path(root.name) / path.relative_to(root)))
        return Response(
            content=buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}_artifacts.zip"'},
        )

    @app.post("/v1/jobs/{job_id}/stop")
    def stop(job_id: str):
        return jobs.stop(job_id)

    @app.post("/v1/teacher_logits")
    def teacher_logits(req: TensorRequest):
        if teacher is None:
            raise HTTPException(status_code=503, detail="No teacher model configured for full-logit worker mode")
        import torch

        with torch.no_grad():
            ids = torch.tensor(req.input_ids, dtype=torch.long)
            logits = teacher(input_ids=ids).logits.detach().cpu()
        if req.response_format == "json_nested":
            return {"format": "nested_float32", "shape": list(logits.shape), "logits": logits.tolist()}
        logits = logits.contiguous().to(torch.float32)
        return Response(
            content=logits.numpy().astype("<f4", copy=False).tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Slimder-Logits-Format": "float32_le",
                "X-Slimder-Logits-Shape": ",".join(str(x) for x in logits.shape),
            },
        )

    return app


def _request_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-slimder-worker-token")


def _artifact_listing(job: dict) -> list[dict]:
    rows = []
    for root_text in job.get("artifact_paths", []):
        root = Path(root_text)
        if root.is_file():
            rows.append({"root": str(root), "files": [{"path": root.name, "bytes": root.stat().st_size}]})
        elif root.is_dir():
            files = [
                {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
                for path in sorted(root.rglob("*"))
                if path.is_file() and not path.is_symlink()
            ]
            rows.append({"root": str(root), "files": files})
        else:
            rows.append({"root": str(root), "missing": True, "files": []})
    return rows

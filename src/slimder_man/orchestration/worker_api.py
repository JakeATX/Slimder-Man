from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class JobRequest(BaseModel):
    config_path: str | None = None
    command: str = "run"


class TensorRequest(BaseModel):
    input_ids: list[list[int]]


def create_worker_app(model_id_or_path: str | None = None) -> FastAPI:
    app = FastAPI(title="Slimder Man Worker")
    jobs: dict[str, dict] = {}
    teacher = None
    if model_id_or_path:
        from transformers import AutoModelForCausalLM

        teacher = AutoModelForCausalLM.from_pretrained(model_id_or_path, trust_remote_code=True)
        teacher.eval()

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
        job_id = str(len(jobs) + 1)
        jobs[job_id] = {"id": job_id, "status": "queued", "request": req.model_dump()}
        return jobs[job_id]

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        return jobs.get(job_id, {"id": job_id, "status": "missing"})

    @app.get("/v1/jobs/{job_id}/logs")
    def logs(job_id: str):
        return {"id": job_id, "logs": []}

    @app.post("/v1/jobs/{job_id}/stop")
    def stop(job_id: str):
        jobs.setdefault(job_id, {"id": job_id})["status"] = "stopped"
        return jobs[job_id]

    @app.post("/v1/teacher_logits")
    def teacher_logits(req: TensorRequest):
        if teacher is None:
            raise HTTPException(status_code=503, detail="No teacher model configured for full-logit worker mode")
        import torch

        with torch.no_grad():
            ids = torch.tensor(req.input_ids, dtype=torch.long)
            logits = teacher(input_ids=ids).logits.detach().cpu()
        return {"format": "nested_float32", "shape": list(logits.shape), "logits": logits.tolist()}

    return app

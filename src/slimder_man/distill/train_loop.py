from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.losses import total_distill_loss
from slimder_man.distill.schedules import cosine_schedule, global_cosine_lr, linear_schedule


def train_tiny_distill(teacher: TinyMoEForCausalLM, student: TinyMoEForCausalLM, cfg: SlimderConfig, output_dir: str | Path, resume: bool = False) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "trainer_state.json"
    resume_model_dir = out_dir / "resume_model"
    start_step = 0
    logs = []
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_step = state.get("global_step", 0)
        logs = list(state.get("logs", []))
        rng_path = out_dir / "rng_state.pt"
        if rng_path.exists():
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            torch.set_rng_state(rng["torch_rng_state"])
            random.setstate(rng["python_random_state"])
            np.random.set_state(rng["numpy_random_state"])
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)
    opt = torch.optim.AdamW(student.parameters(), lr=cfg.training.learning_rate)
    opt_path = out_dir / "optimizer.pt"
    if resume and opt_path.exists():
        opt.load_state_dict(torch.load(opt_path, map_location="cpu"))
    total_steps = cfg.training.train_steps
    teacher.eval()
    student.train()
    for step in range(start_step, total_steps):
        batch = batches[step % len(batches)]
        lr = global_cosine_lr(cfg.training.learning_rate, cfg.training.min_learning_rate, cfg.training.warmup_steps, step, total_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        with torch.no_grad():
            teacher_out = teacher(batch)
        student_out = student(batch)
        lambda_t = linear_schedule(cfg.kd.lambda_schedule.start, cfg.kd.lambda_schedule.end, step, total_steps)
        beta_t = cosine_schedule(cfg.kd.mtp.beta_schedule.start, cfg.kd.mtp.beta_schedule.end, step, total_steps)
        loss, parts = total_distill_loss(
            student_out,
            teacher_out,
            batch,
            lambda_t,
            beta_t,
            cfg.kd.temperature,
            kd_enabled=cfg.kd.enabled,
            mtp_enabled=cfg.kd.mtp.enabled,
        )
        if not torch.isfinite(loss):
            raise ValueError("NaN or Inf distillation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.training.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        row = {"step": step + 1, "loss": float(loss.detach()), "lambda": lambda_t, "beta": beta_t, "lr": lr, **parts}
        logs.append(row)
        torch.save(opt.state_dict(), opt_path)
        student.save_pretrained(resume_model_dir)
        torch.save(
            {
                "torch_rng_state": torch.get_rng_state(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
            },
            out_dir / "rng_state.pt",
        )
        state_path.write_text(
            json.dumps(
                {
                    "global_step": step + 1,
                    "logs": logs,
                    "config": cfg.model_dump(mode="json"),
                    "optimizer_state": str(opt_path),
                    "rng_state": str(out_dir / "rng_state.pt"),
                    "dataloader_position": (step + 1) % len(batches),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    student.save_pretrained(out_dir / "final")
    report = ["# Slimder Man Training Report", ""]
    for row in logs:
        report.append(json.dumps(row, sort_keys=True))
    (out_dir / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"global_step": total_steps, "logs": logs, "checkpoint": str(out_dir / "final")}


def train_causal_lm_distill(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    cfg: SlimderConfig,
    output_dir: str | Path,
    batches: list[torch.Tensor],
    resume: bool = False,
) -> dict:
    """Small generic HF-style distillation loop used by non-tiny smoke fixtures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "trainer_state.json"
    start_step = 0
    logs = []
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_step = state.get("global_step", 0)
        logs = list(state.get("logs", []))
        rng_path = out_dir / "rng_state.pt"
        if rng_path.exists():
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            torch.set_rng_state(rng["torch_rng_state"])
            random.setstate(rng["python_random_state"])
            np.random.set_state(rng["numpy_random_state"])

    opt = torch.optim.AdamW(student.parameters(), lr=cfg.training.learning_rate)
    opt_path = out_dir / "optimizer.pt"
    if resume and opt_path.exists():
        opt.load_state_dict(torch.load(opt_path, map_location="cpu"))

    total_steps = cfg.training.train_steps
    teacher.eval()
    student.train()
    for step in range(start_step, total_steps):
        batch = batches[step % len(batches)]
        lr = global_cosine_lr(cfg.training.learning_rate, cfg.training.min_learning_rate, cfg.training.warmup_steps, step, total_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        with torch.no_grad():
            teacher_out = teacher(input_ids=batch)
        student_out = student(input_ids=batch)
        lambda_t = linear_schedule(cfg.kd.lambda_schedule.start, cfg.kd.lambda_schedule.end, step, total_steps)
        beta_t = cosine_schedule(cfg.kd.mtp.beta_schedule.start, cfg.kd.mtp.beta_schedule.end, step, total_steps)
        loss, parts = total_distill_loss(
            student_out,
            teacher_out,
            batch,
            lambda_t,
            beta_t,
            cfg.kd.temperature,
            kd_enabled=cfg.kd.enabled,
            mtp_enabled=cfg.kd.mtp.enabled and bool(getattr(student_out, "mtp_logits", [])),
        )
        if not torch.isfinite(loss):
            raise ValueError("NaN or Inf distillation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.training.max_grad_norm)
        opt.step()
        opt.zero_grad(set_to_none=True)
        row = {"step": step + 1, "loss": float(loss.detach()), "lambda": lambda_t, "beta": beta_t, "lr": lr, **parts}
        logs.append(row)
        torch.save(opt.state_dict(), opt_path)
        if hasattr(student, "save_pretrained"):
            student.save_pretrained(out_dir / "resume_model")
        torch.save(
            {
                "torch_rng_state": torch.get_rng_state(),
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
            },
            out_dir / "rng_state.pt",
        )
        state_path.write_text(
            json.dumps(
                {
                    "global_step": step + 1,
                    "logs": logs,
                    "config": cfg.model_dump(mode="json"),
                    "optimizer_state": str(opt_path),
                    "rng_state": str(out_dir / "rng_state.pt"),
                    "dataloader_position": (step + 1) % len(batches),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    if hasattr(student, "save_pretrained"):
        student.save_pretrained(out_dir / "final")
    report = ["# Slimder Man Training Report", ""]
    for row in logs:
        report.append(json.dumps(row, sort_keys=True))
    (out_dir / "training_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"global_step": total_steps, "logs": logs, "checkpoint": str(out_dir / "final")}

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from slimder_man.adapters.registry import get_adapter
from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.config.schema import SlimderConfig, load_config
from slimder_man.distill.losses import total_distill_loss
from slimder_man.distill.schedules import cosine_schedule, global_cosine_lr, linear_schedule
from slimder_man.utils.determinism import set_seed


def _concrete_int(value: int | str, fallback: int) -> int:
    return value if isinstance(value, int) else fallback


def _optimizer_steps(cfg: SlimderConfig) -> int:
    if cfg.training.train_steps > 0:
        return cfg.training.train_steps
    global_batch = _concrete_int(cfg.training.global_batch_size, _concrete_int(cfg.training.micro_batch_size, 1))
    tokens_per_step = max(1, global_batch * cfg.training.sequence_length)
    return max(1, int(np.ceil(cfg.training.token_budget / tokens_per_step)))


def _gradient_accumulation_steps(cfg: SlimderConfig, world_size: int = 1) -> int:
    micro = _concrete_int(cfg.training.micro_batch_size, 1)
    global_batch = _concrete_int(cfg.training.global_batch_size, micro * world_size)
    denom = micro * max(1, world_size)
    if global_batch < denom:
        raise ValueError("training.global_batch_size must be >= micro_batch_size * world_size")
    if global_batch % denom != 0:
        raise ValueError("training.global_batch_size must be divisible by micro_batch_size * world_size")
    return max(1, global_batch // denom)


def _microbatch_size(cfg: SlimderConfig) -> int:
    return _concrete_int(cfg.training.micro_batch_size, 1)


def _microbatch_from_samples(batches: list[torch.Tensor], start: int, micro_batch_size: int) -> torch.Tensor:
    if not batches:
        raise ValueError("training requires at least one batch")
    rows = []
    collected = 0
    cursor = start
    while collected < micro_batch_size:
        item = batches[cursor % len(batches)]
        rows.append(item)
        collected += int(item.shape[0])
        cursor += 1
    batch = torch.cat(rows, dim=0)
    return batch[:micro_batch_size]


def _validate_teacher_mode(cfg: SlimderConfig) -> None:
    if cfg.kd.teacher_mode != "online_full_logits":
        raise ValueError(
            f"Local trainer currently supports kd.teacher_mode=online_full_logits only; got {cfg.kd.teacher_mode}. "
            "Use a remote worker or cache-generation path once that backend is configured."
        )


def _mean_parts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}


def _moe_aux_weight(cfg: SlimderConfig) -> float:
    value = cfg.training.moe_aux_loss.weight
    if value == "adapter_default":
        return 0.0
    return float(value)


def train_tiny_distill(
    teacher: TinyMoEForCausalLM,
    student: TinyMoEForCausalLM,
    cfg: SlimderConfig,
    output_dir: str | Path,
    resume: bool = False,
    global_step_offset: int = 0,
    global_total_steps: int | None = None,
) -> dict:
    _validate_teacher_mode(cfg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "trainer_state.json"
    resume_model_dir = out_dir / "resume_model"
    start_step = 0
    logs = []
    if not resume:
        set_seed(cfg.project.seed)
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_step = state.get("stage_step")
        if start_step is None:
            start_step = max(0, state.get("global_step", 0) - global_step_offset)
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
        opt.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))
    total_steps = _optimizer_steps(cfg)
    accum_steps = _gradient_accumulation_steps(cfg)
    micro_batch_size = _microbatch_size(cfg)
    moe_aux_weight = _moe_aux_weight(cfg)
    schedule_total_steps = global_total_steps or total_steps
    teacher.eval()
    student.train()
    for step in range(start_step, total_steps):
        schedule_step = global_step_offset + step
        lr = global_cosine_lr(
            cfg.training.learning_rate,
            cfg.training.min_learning_rate,
            cfg.training.warmup_steps,
            schedule_step,
            schedule_total_steps,
        )
        for group in opt.param_groups:
            group["lr"] = lr
        lambda_t = linear_schedule(cfg.kd.lambda_schedule.start, cfg.kd.lambda_schedule.end, schedule_step, schedule_total_steps)
        beta_t = cosine_schedule(cfg.kd.mtp.beta_schedule.start, cfg.kd.mtp.beta_schedule.end, schedule_step, schedule_total_steps)
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        part_rows: list[dict[str, float]] = []
        for micro_step in range(accum_steps):
            micro_batch = _microbatch_from_samples(
                batches,
                step * accum_steps * micro_batch_size + micro_step * micro_batch_size,
                micro_batch_size,
            )
            with torch.no_grad():
                teacher_out = teacher(micro_batch)
            student_out = student(micro_batch)
            loss, parts = total_distill_loss(
                student_out,
                teacher_out,
                micro_batch,
                lambda_t,
                beta_t,
                cfg.kd.temperature,
                kd_enabled=cfg.kd.enabled,
                mtp_enabled=cfg.kd.mtp.enabled,
                moe_aux_weight=moe_aux_weight,
            )
            if not torch.isfinite(loss):
                raise ValueError("NaN or Inf distillation loss")
            (loss / accum_steps).backward()
            loss_total += float(loss.detach())
            part_rows.append(parts)
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.training.max_grad_norm)
        opt.step()
        parts = _mean_parts(part_rows)
        row = {
            "step": schedule_step + 1,
            "stage_step": step + 1,
            "loss": loss_total / accum_steps,
            "lambda": lambda_t,
            "beta": beta_t,
            "lr": lr,
            "gradient_accumulation_steps": accum_steps,
            "micro_batch_size": micro_batch_size,
            **parts,
        }
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
                    "global_step": schedule_step + 1,
                    "stage_step": step + 1,
                    "global_step_offset": global_step_offset,
                    "global_total_steps": schedule_total_steps,
                    "gradient_accumulation_steps": accum_steps,
                    "micro_batch_size": micro_batch_size,
                    "logs": logs,
                    "config": cfg.model_dump(mode="json"),
                    "optimizer_state": str(opt_path),
                    "rng_state": str(out_dir / "rng_state.pt"),
                    "dataloader_position": ((step + 1) * accum_steps * micro_batch_size) % len(batches),
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
    return {
        "global_step": global_step_offset + total_steps,
        "stage_steps": total_steps,
        "global_total_steps": schedule_total_steps,
        "gradient_accumulation_steps": accum_steps,
        "micro_batch_size": micro_batch_size,
        "logs": logs,
        "checkpoint": str(out_dir / "final"),
    }


def train_causal_lm_distill(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    cfg: SlimderConfig,
    output_dir: str | Path,
    batches: list[torch.Tensor],
    resume: bool = False,
    global_step_offset: int = 0,
    global_total_steps: int | None = None,
) -> dict:
    """Small generic HF-style distillation loop used by non-tiny smoke fixtures."""
    _validate_teacher_mode(cfg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "trainer_state.json"
    start_step = 0
    logs = []
    if not resume:
        set_seed(cfg.project.seed)
    if resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_step = state.get("stage_step")
        if start_step is None:
            start_step = max(0, state.get("global_step", 0) - global_step_offset)
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
        opt.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=True))

    total_steps = _optimizer_steps(cfg)
    accum_steps = _gradient_accumulation_steps(cfg)
    micro_batch_size = _microbatch_size(cfg)
    moe_aux_weight = _moe_aux_weight(cfg)
    schedule_total_steps = global_total_steps or total_steps
    teacher.eval()
    student.train()
    for step in range(start_step, total_steps):
        schedule_step = global_step_offset + step
        lr = global_cosine_lr(
            cfg.training.learning_rate,
            cfg.training.min_learning_rate,
            cfg.training.warmup_steps,
            schedule_step,
            schedule_total_steps,
        )
        for group in opt.param_groups:
            group["lr"] = lr
        lambda_t = linear_schedule(cfg.kd.lambda_schedule.start, cfg.kd.lambda_schedule.end, schedule_step, schedule_total_steps)
        beta_t = cosine_schedule(cfg.kd.mtp.beta_schedule.start, cfg.kd.mtp.beta_schedule.end, schedule_step, schedule_total_steps)
        opt.zero_grad(set_to_none=True)
        loss_total = 0.0
        part_rows: list[dict[str, float]] = []
        for micro_step in range(accum_steps):
            micro_batch = _microbatch_from_samples(
                batches,
                step * accum_steps * micro_batch_size + micro_step * micro_batch_size,
                micro_batch_size,
            )
            with torch.no_grad():
                teacher_out = teacher(input_ids=micro_batch)
            student_out = student(input_ids=micro_batch)
            loss, parts = total_distill_loss(
                student_out,
                teacher_out,
                micro_batch,
                lambda_t,
                beta_t,
                cfg.kd.temperature,
                kd_enabled=cfg.kd.enabled,
                mtp_enabled=cfg.kd.mtp.enabled and bool(getattr(student_out, "mtp_logits", [])),
                moe_aux_weight=moe_aux_weight,
            )
            if not torch.isfinite(loss):
                raise ValueError("NaN or Inf distillation loss")
            (loss / accum_steps).backward()
            loss_total += float(loss.detach())
            part_rows.append(parts)
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.training.max_grad_norm)
        opt.step()
        parts = _mean_parts(part_rows)
        row = {
            "step": schedule_step + 1,
            "stage_step": step + 1,
            "loss": loss_total / accum_steps,
            "lambda": lambda_t,
            "beta": beta_t,
            "lr": lr,
            "gradient_accumulation_steps": accum_steps,
            "micro_batch_size": micro_batch_size,
            **parts,
        }
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
                    "global_step": schedule_step + 1,
                    "stage_step": step + 1,
                    "global_step_offset": global_step_offset,
                    "global_total_steps": schedule_total_steps,
                    "gradient_accumulation_steps": accum_steps,
                    "micro_batch_size": micro_batch_size,
                    "logs": logs,
                    "config": cfg.model_dump(mode="json"),
                    "optimizer_state": str(opt_path),
                    "rng_state": str(out_dir / "rng_state.pt"),
                    "dataloader_position": ((step + 1) * accum_steps * micro_batch_size) % len(batches),
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
    return {
        "global_step": global_step_offset + total_steps,
        "stage_steps": total_steps,
        "global_total_steps": schedule_total_steps,
        "gradient_accumulation_steps": accum_steps,
        "micro_batch_size": micro_batch_size,
        "logs": logs,
        "checkpoint": str(out_dir / "final"),
    }


def run_train_loop_entrypoint(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    resume: bool = False,
) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg.project.seed)
    out_dir = Path(output_dir or Path(cfg.project.output_dir) / "training")
    if cfg.teacher.load_mode == "tiny":
        teacher = TinyMoEForCausalLM()
        if checkpoint:
            student = TinyMoEForCausalLM.from_pretrained(checkpoint)
        elif resume and (out_dir / "resume_model").exists():
            student = TinyMoEForCausalLM.from_pretrained(out_dir / "resume_model")
        else:
            student = TinyMoEForCausalLM()
        result = train_tiny_distill(teacher, student, cfg, out_dir, resume=resume)
        return {"entrypoint": "slimder_man.distill.train_loop", "mode": "tiny", **result}

    if cfg.teacher.model_id_or_path != "dummy-hf-moe":
        if not cfg.runtime.local.allow_full_model_run:
            raise ValueError(
                "train_loop entrypoint for arbitrary Transformers checkpoints requires runtime.local.allow_full_model_run=true "
                "to avoid accidental full-model downloads. Pass --checkpoint with a compressed student checkpoint and opt in explicitly."
            )
        init_checkpoint = Path(checkpoint) if checkpoint else None
        if resume and (out_dir / "resume_model").exists():
            init_checkpoint = out_dir / "resume_model"
        if init_checkpoint is None:
            raise ValueError("--checkpoint is required for arbitrary Transformers distillation entrypoint runs")
        teacher = _load_entrypoint_transformers_model(cfg)
        student = _load_entrypoint_transformers_checkpoint(cfg, init_checkpoint)
        tokenizer = _load_entrypoint_transformers_tokenizer(cfg)
        arch = get_adapter(student).describe_architecture(student)
        batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=arch.vocab_size, tokenizer=tokenizer)
        result = train_causal_lm_distill(teacher, student, cfg, out_dir, batches, resume=resume)
        return {"entrypoint": "slimder_man.distill.train_loop", "mode": "transformers", "student_checkpoint": str(init_checkpoint), **result}

    from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM, DummyTokenizer

    teacher = DummyHfMoeForCausalLM()
    if checkpoint:
        student = DummyHfMoeForCausalLM.from_pretrained(checkpoint)
    elif resume and (out_dir / "resume_model").exists():
        student = DummyHfMoeForCausalLM.from_pretrained(out_dir / "resume_model")
    else:
        student = DummyHfMoeForCausalLM()
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size, tokenizer=DummyTokenizer())
    result = train_causal_lm_distill(teacher, student, cfg, out_dir, batches, resume=resume)
    return {"entrypoint": "slimder_man.distill.train_loop", "mode": "dummy_hf_moe", **result}


def _load_entrypoint_transformers_model(cfg: SlimderConfig):
    from transformers import AutoModelForCausalLM

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return AutoModelForCausalLM.from_pretrained(
        cfg.teacher.model_id_or_path,
        revision=cfg.teacher.revision,
        trust_remote_code=cfg.teacher.trust_remote_code,
        torch_dtype=dtype_map.get(cfg.teacher.dtype, torch.float32),
        device_map=cfg.teacher.device_map,
    )


def _load_entrypoint_transformers_checkpoint(cfg: SlimderConfig, checkpoint: str | Path):
    from transformers import AutoModelForCausalLM

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=cfg.teacher.trust_remote_code,
        torch_dtype=dtype_map.get(cfg.teacher.dtype, torch.float32),
        device_map=cfg.teacher.device_map,
    )


def _load_entrypoint_transformers_tokenizer(cfg: SlimderConfig):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.teacher.model_id_or_path,
        revision=cfg.teacher.revision,
        trust_remote_code=cfg.teacher.trust_remote_code,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Slimder Man distillation training.")
    parser.add_argument("--config", required=True, help="Slimder YAML config.")
    parser.add_argument("--output-dir", default=None, help="Training output directory. Defaults to project.output_dir/training.")
    parser.add_argument("--checkpoint", default=None, help="Optional student checkpoint to resume/init from.")
    parser.add_argument("--resume", action="store_true", help="Resume optimizer/RNG/model state from output-dir.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    result = run_train_loop_entrypoint(args.config, args.output_dir, args.checkpoint, resume=args.resume)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(result)


if __name__ == "__main__":
    main()

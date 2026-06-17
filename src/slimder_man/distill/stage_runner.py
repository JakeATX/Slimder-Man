from __future__ import annotations

from pathlib import Path
from typing import Callable

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.calibration.collectors import CalibrationResult, collect_tiny_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import compress_tiny_model
from slimder_man.compression.planner import StagePlan, progressive_plan
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.train_loop import train_tiny_distill

CalibrationFn = Callable[[TinyMoEForCausalLM, SlimderConfig], CalibrationResult]
CompressFn = Callable[[TinyMoEForCausalLM, SlimderConfig, CalibrationResult, Path], tuple[TinyMoEForCausalLM, dict]]
TrainFn = Callable[[TinyMoEForCausalLM, TinyMoEForCausalLM, SlimderConfig, Path], dict]


def stage_plans_for_tiny(model: TinyMoEForCausalLM, cfg: SlimderConfig) -> list[StagePlan]:
    return progressive_plan(
        cfg.progressive.schedule,
        cfg.progressive.stages,
        cfg.training.token_budget,
        cfg.progressive.token_split,
        len(model.layers),
        cfg.compression.target.remove_last_n_layers,
        model.config.hidden_size,
        cfg.compression.target.hidden_size,
        cfg.compression.width.hidden_size_multiple,
        cfg.compression.target.routed_experts,
        cfg.compression.target.routed_top_k,
    )


def config_for_stage(base_cfg: SlimderConfig, plan: StagePlan, teacher: TinyMoEForCausalLM) -> SlimderConfig:
    target = base_cfg.compression.target.model_copy(
        update={
            "remove_last_n_layers": plan.remove_last_n_layers,
            "hidden_size": plan.hidden_size,
            "routed_experts": plan.routed_experts if plan.routed_experts is not None else teacher.config.num_routed_experts,
            "routed_top_k": plan.top_k if plan.top_k is not None else teacher.config.top_k,
        }
    )
    compression = base_cfg.compression.model_copy(update={"target": target})
    training = base_cfg.training.model_copy(update={"token_budget": plan.tokens})
    return base_cfg.model_copy(update={"compression": compression, "training": training})


def _default_calibrate(model: TinyMoEForCausalLM, cfg: SlimderConfig) -> CalibrationResult:
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=model.config.vocab_size)
    return collect_tiny_calibration(model, batches)


def run_tiny_progressive_stages(
    teacher: TinyMoEForCausalLM,
    cfg: SlimderConfig,
    output_dir: str | Path,
    calibrate_fn: CalibrationFn = _default_calibrate,
    compress_fn: CompressFn = compress_tiny_model,
    train_fn: TrainFn = train_tiny_distill,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = []
    plans = stage_plans_for_tiny(teacher, cfg)
    for plan in plans:
        stage_cfg = config_for_stage(cfg, plan, teacher)
        stage_dir = out_dir / f"stage_{plan.stage}"
        cal = calibrate_fn(teacher, stage_cfg)
        student, manifest = compress_fn(teacher, stage_cfg, cal, stage_dir / "compressed")
        train = train_fn(teacher, student, stage_cfg, stage_dir / "training")
        stages.append(
            {
                "stage": plan.stage,
                "tokens": plan.tokens,
                "checkpoint": str(stage_dir / "compressed"),
                "training": train,
                "manifest": manifest,
            }
        )
    return {
        "stages": stages,
        "final_checkpoint": stages[-1]["checkpoint"] if stages else None,
        "final_training_checkpoint": stages[-1]["training"].get("checkpoint") if stages else None,
    }

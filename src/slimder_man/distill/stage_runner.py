from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.analyze.architecture import describe_model
from slimder_man.calibration.artifacts import write_calibration_artifacts
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
    source_config_path: str | Path | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = []
    plans = stage_plans_for_tiny(teacher, cfg)
    previous_checkpoint: str | None = None
    arch = describe_model(teacher)
    for plan in plans:
        stage_cfg = config_for_stage(cfg, plan, teacher)
        stage_dir = out_dir / f"stage_{plan.stage}"
        analysis_dir = stage_dir / "analysis"
        _, source_manifest = sample_calibration_tokens(stage_cfg.calibration, vocab_size=teacher.config.vocab_size)
        cal = calibrate_fn(teacher, stage_cfg)
        write_calibration_artifacts(analysis_dir, stage_cfg, cal, source_manifest, arch)
        stage_provenance = {
            "stage": plan.stage,
            "total_stages": len(plans),
            "token_split": cfg.progressive.token_split,
            "stage_token_budget": plan.tokens,
            "source": "teacher" if previous_checkpoint is None else "previous_stage_checkpoint",
            "previous_checkpoint": previous_checkpoint,
            "final_stage": plan.stage == len(plans),
        }
        compress_kwargs = {
            "calibration_manifest_path": analysis_dir / "calibration_manifest.json",
            "source_config_path": source_config_path,
            "stage_provenance": stage_provenance,
        }
        signature = inspect.signature(compress_fn)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        supported_kwargs = {
            key: value
            for key, value in compress_kwargs.items()
            if accepts_kwargs or key in signature.parameters
        }
        student, manifest = compress_fn(teacher, stage_cfg, cal, stage_dir / "compressed", **supported_kwargs)
        train = train_fn(teacher, student, stage_cfg, stage_dir / "training")
        previous_checkpoint = str(stage_dir / "compressed")
        stages.append(
            {
                "stage": plan.stage,
                "tokens": plan.tokens,
                "analysis": str(analysis_dir),
                "checkpoint": str(stage_dir / "compressed"),
                "training": train,
                "manifest": manifest,
                "stage_provenance": stage_provenance,
            }
        )
    return {
        "stages": stages,
        "final_checkpoint": stages[-1]["checkpoint"] if stages else None,
        "final_training_checkpoint": stages[-1]["training"].get("checkpoint") if stages else None,
    }

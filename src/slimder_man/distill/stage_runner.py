from __future__ import annotations

import inspect
from pathlib import Path
import math
from typing import Callable

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.adapters.registry import get_adapter
from slimder_man.analyze.architecture import describe_model
from slimder_man.calibration.artifacts import write_calibration_artifacts
from slimder_man.calibration.collectors import CalibrationResult, collect_calibration, collect_tiny_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import compress_model, compress_tiny_model
from slimder_man.compression.planner import StagePlan, progressive_plan
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.train_loop import train_causal_lm_distill, train_tiny_distill

CalibrationFn = Callable[[TinyMoEForCausalLM, SlimderConfig], CalibrationResult]
CompressFn = Callable[[TinyMoEForCausalLM, SlimderConfig, CalibrationResult, Path], tuple[TinyMoEForCausalLM, dict]]
TrainFn = Callable[..., dict]
GenericLoadModelFn = Callable[[], object]
GenericLoadCheckpointFn = Callable[[Path], object]
GenericCalibrateFn = Callable[[object, SlimderConfig], tuple[CalibrationResult, dict]]
GenericCompressFn = Callable[..., tuple[object, dict]]
GenericTrainFn = Callable[..., dict]


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


def stage_plans_for_architecture(architecture: dict, cfg: SlimderConfig) -> list[StagePlan]:
    return progressive_plan(
        cfg.progressive.schedule,
        cfg.progressive.stages,
        cfg.training.token_budget,
        cfg.progressive.token_split,
        int(architecture["num_layers"]),
        cfg.compression.target.remove_last_n_layers,
        int(architecture["hidden_size"]),
        cfg.compression.target.hidden_size,
        cfg.compression.width.hidden_size_multiple,
        cfg.compression.target.routed_experts,
        cfg.compression.target.routed_top_k,
    )


def config_for_stage(
    base_cfg: SlimderConfig,
    plan: StagePlan,
    teacher: TinyMoEForCausalLM,
    source_layers: int | None = None,
) -> SlimderConfig:
    teacher_layers = len(teacher.layers)
    current_layers = source_layers if source_layers is not None else teacher_layers
    desired_layers = max(1, teacher_layers - plan.remove_last_n_layers)
    remove_from_source = max(0, current_layers - desired_layers)
    target = base_cfg.compression.target.model_copy(
        update={
            "remove_last_n_layers": remove_from_source,
            "hidden_size": plan.hidden_size,
            "routed_experts": plan.routed_experts if plan.routed_experts is not None else teacher.config.num_routed_experts,
            "routed_top_k": plan.top_k if plan.top_k is not None else teacher.config.top_k,
        }
    )
    compression = base_cfg.compression.model_copy(update={"target": target})
    training = base_cfg.training.model_copy(update={"token_budget": plan.tokens})
    return base_cfg.model_copy(update={"compression": compression, "training": training})


def config_for_generic_stage(
    base_cfg: SlimderConfig,
    plan: StagePlan,
    teacher_architecture: dict,
    source_architecture: dict | None = None,
) -> SlimderConfig:
    teacher_layers = int(teacher_architecture["num_layers"])
    current_layers = int((source_architecture or teacher_architecture)["num_layers"])
    desired_layers = max(1, teacher_layers - plan.remove_last_n_layers)
    remove_from_source = max(0, current_layers - desired_layers)
    source_experts = _source_routed_experts(source_architecture or teacher_architecture)
    source_top_k = _source_top_k(source_architecture or teacher_architecture)
    target = base_cfg.compression.target.model_copy(
        update={
            "remove_last_n_layers": remove_from_source,
            "hidden_size": plan.hidden_size,
            "routed_experts": plan.routed_experts if plan.routed_experts is not None else source_experts,
            "routed_top_k": plan.top_k if plan.top_k is not None else source_top_k,
        }
    )
    compression = base_cfg.compression.model_copy(update={"target": target})
    training = base_cfg.training.model_copy(update={"token_budget": plan.tokens})
    return base_cfg.model_copy(update={"compression": compression, "training": training})


def _concrete_int(value: int | str, fallback: int) -> int:
    return value if isinstance(value, int) else fallback


def _stage_train_steps(base_cfg: SlimderConfig, plan: StagePlan) -> int:
    if plan.tokens <= 0:
        return 0
    batch = _concrete_int(base_cfg.training.global_batch_size, _concrete_int(base_cfg.training.micro_batch_size, 1))
    seq = max(1, base_cfg.training.sequence_length)
    tokens_per_step = max(1, batch * seq)
    return max(1, math.ceil(plan.tokens / tokens_per_step))


def _default_calibrate(model: TinyMoEForCausalLM, cfg: SlimderConfig) -> CalibrationResult:
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=model.config.vocab_size)
    return collect_tiny_calibration(model, batches)


def _source_routed_experts(architecture: dict) -> int:
    moe_layers = architecture.get("moe_layers") or []
    if moe_layers:
        return max(int(layer["num_routed_experts"]) for layer in moe_layers)
    return int(architecture.get("routed_experts") or architecture.get("num_routed_experts") or 1)


def _source_top_k(architecture: dict) -> int:
    moe_layers = architecture.get("moe_layers") or []
    if moe_layers:
        return max(int(layer["top_k"]) for layer in moe_layers)
    return int(architecture.get("routed_top_k") or architecture.get("top_k") or 1)


def _default_generic_calibrate(model: object, cfg: SlimderConfig, tokenizer=None) -> tuple[CalibrationResult, dict]:
    arch = describe_model(model)
    batches, source_manifest = sample_calibration_tokens(cfg.calibration, vocab_size=int(arch["vocab_size"]), tokenizer=tokenizer)
    return collect_calibration(model, batches, get_adapter(model)), source_manifest


def _call_with_supported_kwargs(fn: Callable, *args, **kwargs):
    signature = inspect.signature(fn)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if accepts_kwargs or key in signature.parameters
    }
    return fn(*args, **supported_kwargs)


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
    stage_steps = [_stage_train_steps(cfg, plan) for plan in plans]
    global_total_steps = sum(stage_steps)
    global_step_offset = 0
    previous_checkpoint: str | None = None
    source_model = teacher
    teacher_layers = len(teacher.layers)
    source_layers = teacher_layers
    for plan, train_steps in zip(plans, stage_steps, strict=True):
        stage_cfg = config_for_stage(cfg, plan, teacher, source_layers=source_layers)
        stage_cfg = stage_cfg.model_copy(update={"training": stage_cfg.training.model_copy(update={"train_steps": train_steps})})
        stage_dir = out_dir / f"stage_{plan.stage}"
        analysis_dir = stage_dir / "analysis"
        arch = describe_model(source_model)
        _, source_manifest = sample_calibration_tokens(stage_cfg.calibration, vocab_size=source_model.config.vocab_size)
        cal = calibrate_fn(source_model, stage_cfg)
        write_calibration_artifacts(analysis_dir, stage_cfg, cal, source_manifest, arch)
        cumulative_target_layers = max(1, teacher_layers - plan.remove_last_n_layers)
        stage_provenance = {
            "stage": plan.stage,
            "total_stages": len(plans),
            "token_split": cfg.progressive.token_split,
            "stage_token_budget": plan.tokens,
            "source": "teacher" if previous_checkpoint is None else "previous_stage_checkpoint",
            "previous_checkpoint": previous_checkpoint,
            "final_stage": plan.stage == len(plans),
            "cumulative_target": {
                "remove_last_n_layers": plan.remove_last_n_layers,
                "hidden_size": plan.hidden_size,
                "layers": cumulative_target_layers,
                "routed_experts": plan.routed_experts,
                "top_k": plan.top_k,
            },
        }
        compress_kwargs = {
            "calibration_manifest_path": analysis_dir / "calibration_manifest.json",
            "source_config_path": source_config_path,
            "stage_provenance": stage_provenance,
        }
        student, manifest = _call_with_supported_kwargs(compress_fn, source_model, stage_cfg, cal, stage_dir / "compressed", **compress_kwargs)
        train_kwargs = {
            "global_step_offset": global_step_offset,
            "global_total_steps": global_total_steps,
        }
        train = _call_with_supported_kwargs(train_fn, teacher, student, stage_cfg, stage_dir / "training", **train_kwargs)
        global_step_offset += train_steps
        train_checkpoint = train.get("checkpoint")
        previous_checkpoint = str(train_checkpoint or stage_dir / "compressed")
        if train_checkpoint and Path(train_checkpoint).exists():
            source_model = TinyMoEForCausalLM.from_pretrained(train_checkpoint)
        else:
            source_model = student
        source_layers = cumulative_target_layers
        stages.append(
            {
                "stage": plan.stage,
                "tokens": plan.tokens,
                "train_steps": train_steps,
                "global_step_start": global_step_offset - train_steps,
                "global_step_end": global_step_offset,
                "analysis": str(analysis_dir),
                "checkpoint": str(stage_dir / "compressed"),
                "training": train,
                "manifest": manifest,
                "stage_provenance": stage_provenance,
            }
        )
    return {
        "stages": stages,
        "global_total_steps": global_total_steps,
        "final_checkpoint": stages[-1]["checkpoint"] if stages else None,
        "final_training_checkpoint": stages[-1]["training"].get("checkpoint") if stages else None,
    }


def run_generic_progressive_stages(
    teacher: object,
    cfg: SlimderConfig,
    output_dir: str | Path,
    *,
    tokenizer=None,
    load_teacher_fn: GenericLoadModelFn | None = None,
    load_checkpoint_fn: GenericLoadCheckpointFn | None = None,
    calibrate_fn: GenericCalibrateFn | None = None,
    compress_fn: GenericCompressFn = compress_model,
    train_fn: GenericTrainFn = train_causal_lm_distill,
    source_config_path: str | Path | None = None,
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_arch = describe_model(teacher)
    plans = stage_plans_for_architecture(teacher_arch, cfg)
    stage_steps = [_stage_train_steps(cfg, plan) for plan in plans]
    global_total_steps = sum(stage_steps)
    global_step_offset = 0
    previous_checkpoint: str | None = None
    source_model = teacher
    stages = []
    for plan, train_steps in zip(plans, stage_steps, strict=True):
        source_arch = describe_model(source_model)
        stage_cfg = config_for_generic_stage(cfg, plan, teacher_arch, source_arch)
        stage_cfg = stage_cfg.model_copy(update={"training": stage_cfg.training.model_copy(update={"train_steps": train_steps})})
        stage_dir = out_dir / f"stage_{plan.stage}"
        analysis_dir = stage_dir / "analysis"
        if calibrate_fn is None:
            cal, source_manifest = _default_generic_calibrate(source_model, stage_cfg, tokenizer=tokenizer)
        else:
            cal, source_manifest = calibrate_fn(source_model, stage_cfg)
        write_calibration_artifacts(analysis_dir, stage_cfg, cal, source_manifest, source_arch)
        cumulative_target_layers = max(1, int(teacher_arch["num_layers"]) - plan.remove_last_n_layers)
        stage_provenance = {
            "stage": plan.stage,
            "total_stages": len(plans),
            "token_split": cfg.progressive.token_split,
            "stage_token_budget": plan.tokens,
            "source": "teacher" if previous_checkpoint is None else "previous_stage_checkpoint",
            "previous_checkpoint": previous_checkpoint,
            "final_stage": plan.stage == len(plans),
            "cumulative_target": {
                "remove_last_n_layers": plan.remove_last_n_layers,
                "hidden_size": plan.hidden_size,
                "layers": cumulative_target_layers,
                "routed_experts": plan.routed_experts,
                "top_k": plan.top_k,
            },
        }
        adapter = get_adapter(source_model)
        student, manifest = _call_with_supported_kwargs(
            compress_fn,
            source_model,
            stage_cfg,
            cal,
            adapter=adapter,
            output_dir=stage_dir / "compressed",
            tokenizer=tokenizer,
            calibration_manifest_path=analysis_dir / "calibration_manifest.json",
            source_config_path=source_config_path,
            stage_provenance=stage_provenance,
        )
        train_batches, _ = sample_calibration_tokens(stage_cfg.calibration, vocab_size=int(source_arch["vocab_size"]), tokenizer=tokenizer)
        teacher_for_distill = load_teacher_fn() if load_teacher_fn is not None else teacher
        train = _call_with_supported_kwargs(
            train_fn,
            teacher_for_distill,
            student,
            stage_cfg,
            stage_dir / "training",
            train_batches,
            resume=False,
            global_step_offset=global_step_offset,
            global_total_steps=global_total_steps,
        )
        global_step_offset += train_steps
        train_checkpoint = train.get("checkpoint")
        previous_checkpoint = str(train_checkpoint or stage_dir / "compressed")
        if train_checkpoint and load_checkpoint_fn is not None and Path(train_checkpoint).exists():
            source_model = load_checkpoint_fn(Path(train_checkpoint))
        else:
            source_model = student
        stages.append(
            {
                "stage": plan.stage,
                "tokens": plan.tokens,
                "train_steps": train_steps,
                "global_step_start": global_step_offset - train_steps,
                "global_step_end": global_step_offset,
                "analysis": str(analysis_dir),
                "checkpoint": str(stage_dir / "compressed"),
                "training": train,
                "manifest": manifest,
                "stage_provenance": stage_provenance,
            }
        )
    return {
        "stages": stages,
        "global_total_steps": global_total_steps,
        "final_checkpoint": stages[-1]["checkpoint"] if stages else None,
        "final_training_checkpoint": stages[-1]["training"].get("checkpoint") if stages else None,
    }

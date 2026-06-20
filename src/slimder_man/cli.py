from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

import typer
import torch
from rich import print

from slimder_man.adapters.registry import get_adapter
from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.analyze.architecture import describe_model
from slimder_man.analyze.recommender import recommend
from slimder_man.analyze.reports import write_analysis_report
from slimder_man.calibration.artifacts import write_calibration_artifacts
from slimder_man.calibration.collectors import collect_calibration, collect_tiny_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import TOKENIZER_ARTIFACT_NAMES, compress_model, compress_tiny_model
from slimder_man.compression.manifests import load_manifest
from slimder_man.compression.validate import validate_tiny_model
from slimder_man.config.defaults import tiny_default_config
from slimder_man.config.schema import SlimderConfig, load_config, save_config
from slimder_man.distill.train_loop import train_causal_lm_distill, train_tiny_distill
from slimder_man.distill.stage_runner import run_tiny_progressive_stages
from slimder_man.eval.perplexity import causal_lm_perplexity, tiny_perplexity
from slimder_man.orchestration.local import local_dry_run_commands
from slimder_man.orchestration.skypilot import SkyPilotRunner, skypilot_yaml
from slimder_man.orchestration.ssh import SSHRunner, ssh_dry_run_commands
from slimder_man.quant.export import write_quant_export_manifest
from slimder_man.quant.fake_backend import fake_quantize_model
from slimder_man.ui.app import create_app
from slimder_man.utils.hashing import sha256_file
from slimder_man.utils.json import read_json, write_json
from slimder_man.utils.determinism import set_seed

app = typer.Typer(no_args_is_help=True)


def _echo(data: dict, as_json: bool) -> None:
    if as_json:
        typer.echo(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(data)


def _load_or_default(path: Optional[Path]) -> SlimderConfig:
    return load_config(path) if path else tiny_default_config()


def _load_cli_config(path: Path) -> SlimderConfig:
    cfg = load_config(path)
    dataset_path = cfg.calibration.dataset.path
    if dataset_path:
        raw_path = Path(dataset_path)
        candidate = (path.parent / raw_path).resolve()
        if not raw_path.is_absolute() and candidate.exists():
            cfg.calibration.dataset.path = str(candidate)
    return cfg


def _resolve_config_path(config: Path | None, config_option: Path | None = None) -> Path:
    resolved = config_option or config
    if resolved is None:
        raise typer.BadParameter("A config path is required")
    return resolved


def _resolve_path_arg(value: Path | None, option: Path | None, label: str) -> Path:
    resolved = option or value
    if resolved is None:
        raise typer.BadParameter(f"{label} path is required")
    return resolved


def _tiny_teacher(cfg: SlimderConfig) -> TinyMoEForCausalLM:
    if cfg.teacher.load_mode != "tiny":
        raise typer.BadParameter("Tiny-only operation requested for a non-tiny teacher")
    return TinyMoEForCausalLM()


def _load_transformers_model(cfg: SlimderConfig):
    if cfg.teacher.model_id_or_path == "dummy-hf-moe":
        from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM

        return DummyHfMoeForCausalLM()
    from transformers import AutoModelForCausalLM
    import torch

    dtype_map = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}
    return AutoModelForCausalLM.from_pretrained(
        cfg.teacher.model_id_or_path,
        revision=cfg.teacher.revision,
        trust_remote_code=cfg.teacher.trust_remote_code,
        torch_dtype=dtype_map.get(cfg.teacher.dtype, torch.float32),
        device_map=cfg.teacher.device_map,
    )


def _load_transformers_checkpoint(cfg: SlimderConfig, checkpoint: Path):
    if cfg.teacher.model_id_or_path == "dummy-hf-moe":
        from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM

        return DummyHfMoeForCausalLM.from_pretrained(checkpoint)
    from transformers import AutoModelForCausalLM
    import torch

    dtype_map = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}
    return AutoModelForCausalLM.from_pretrained(
        checkpoint,
        trust_remote_code=cfg.teacher.trust_remote_code,
        torch_dtype=dtype_map.get(cfg.teacher.dtype, torch.float32),
        device_map=cfg.teacher.device_map,
    )


def _load_transformers_tokenizer(cfg: SlimderConfig):
    if cfg.teacher.model_id_or_path == "dummy-hf-moe":
        from slimder_man.adapters.hf_dummy import DummyTokenizer

        return DummyTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.teacher.model_id_or_path,
        revision=cfg.teacher.revision,
        trust_remote_code=cfg.teacher.trust_remote_code,
    )


def _load_model(cfg: SlimderConfig):
    return _tiny_teacher(cfg) if cfg.teacher.load_mode == "tiny" else _load_transformers_model(cfg)


def _load_tokenizer(cfg: SlimderConfig):
    return None if cfg.teacher.load_mode == "tiny" else _load_transformers_tokenizer(cfg)


def _checkpoint_kind(checkpoint: Path) -> str:
    if (checkpoint / "model.pt").exists():
        return "tiny"
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        raise ValueError(f"Checkpoint is missing config.json: {checkpoint}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Checkpoint config.json is not valid JSON: {checkpoint}") from exc
    if config.get("model_type") == "dummy_hf_moe":
        return "dummy_hf_moe"
    if any((checkpoint / name).exists() for name in ("model.safetensors", "pytorch_model.bin")) or any(checkpoint.glob("model-*.safetensors")):
        return "transformers"
    raise ValueError(f"Unsupported checkpoint format: {checkpoint}")


def _load_checkpoint_auto(checkpoint: Path):
    kind = _checkpoint_kind(checkpoint)
    if kind == "tiny":
        return TinyMoEForCausalLM.from_pretrained(checkpoint), kind
    if kind == "dummy_hf_moe":
        from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM

        return DummyHfMoeForCausalLM.from_pretrained(checkpoint), kind
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True), kind


def _save_checkpoint_auto(model, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(model, TinyMoEForCausalLM):
        model.save_pretrained(output_dir)
        return
    if not hasattr(model, "save_pretrained"):
        raise ValueError("Loaded checkpoint model does not expose save_pretrained")
    try:
        model.save_pretrained(output_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(output_dir)


def _copy_checkpoint_sidecars(source: Path, out: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    names = set(TOKENIZER_ARTIFACT_NAMES) | {"compression_manifest.json"}
    for path in source.rglob("*"):
        if not path.is_file() or path.name not in names:
            continue
        rel = path.relative_to(source)
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        copied[rel.as_posix()] = sha256_file(target)
    return copied


def _copy_and_rewrite_calibration_references(out: Path) -> dict[str, str]:
    manifest_path = out / "compression_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    calibration = manifest.get("calibration_artifacts")
    if not calibration:
        return {}
    target_dir = out / "calibration_artifacts"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    source_manifest = Path(calibration["manifest_path"])
    if not source_manifest.exists():
        raise ValueError(f"Cannot consolidate missing calibration manifest: {source_manifest}")
    target_manifest = target_dir / "calibration_manifest.json"
    shutil.copy2(source_manifest, target_manifest)
    calibration["manifest_path"] = str(target_manifest.resolve())
    calibration["manifest_sha256"] = sha256_file(target_manifest)
    calibration["analysis_dir"] = str(target_dir.resolve())
    copied[target_manifest.relative_to(out).as_posix()] = sha256_file(target_manifest)

    rewritten_by_name: dict[str, dict] = {}
    for name, artifact in calibration.get("artifacts", {}).items():
        source_artifact = Path(artifact["path"])
        if not source_artifact.exists():
            raise ValueError(f"Cannot consolidate missing calibration artifact: {source_artifact}")
        target_artifact = target_dir / name
        shutil.copy2(source_artifact, target_artifact)
        rewritten = {
            **artifact,
            "path": str(target_artifact.resolve()),
            "sha256": sha256_file(target_artifact),
        }
        rewritten_by_name[name] = rewritten
        copied[target_artifact.relative_to(out).as_posix()] = rewritten["sha256"]
    calibration["artifacts"] = rewritten_by_name

    for layer in manifest.get("experts", {}).get("layers", []):
        for key in ("score_artifact", "similarity_artifact"):
            artifact = layer.get(key)
            if not artifact:
                continue
            name = Path(artifact["path"]).name
            if name in rewritten_by_name:
                layer[key] = {**artifact, **rewritten_by_name[name]}
    manifest["calibration_artifacts"] = calibration
    write_json(manifest_path, manifest)
    copied["compression_manifest.json"] = sha256_file(manifest_path)
    return copied


def _forward_validation_errors(model, kind: str) -> list[str]:
    errors: list[str] = []
    try:
        arch = describe_model(model)
        vocab_size = int(arch["vocab_size"])
        input_ids = sample_calibration_tokens(
            SlimderConfig(calibration={"sample_count": 1, "sequence_length": 8}).calibration,
            vocab_size=vocab_size,
        )[0][0]
        with torch.no_grad():
            out = model(input_ids, labels=input_ids) if kind == "tiny" else model(input_ids=input_ids, labels=input_ids)
        logits = out.logits
        if any(dim <= 0 for dim in logits.shape):
            errors.append("logits contain a zero dimension")
        if not torch.isfinite(logits).all():
            errors.append("logits are not finite")
        if out.loss is None:
            errors.append("loss is missing")
        elif not torch.isfinite(out.loss):
            errors.append("loss is not finite")
        for name, tensor in model.state_dict().items():
            if any(dim <= 0 for dim in tensor.shape):
                errors.append(f"tensor has zero dimension: {name}")
                break
    except Exception as exc:
        errors.append(str(exc))
    return errors


def _calibration_artifact_validation_errors(manifest_data: dict | None) -> list[str]:
    if not manifest_data or not manifest_data.get("calibration_artifacts"):
        return []
    errors: list[str] = []
    calibration = manifest_data["calibration_artifacts"]
    manifest_path = Path(calibration["manifest_path"])
    if not manifest_path.exists():
        errors.append(f"calibration manifest missing: {manifest_path}")
    elif sha256_file(manifest_path) != calibration["manifest_sha256"]:
        errors.append(f"calibration manifest hash mismatch: {manifest_path}")
    for artifact in calibration.get("artifacts", {}).values():
        path = Path(artifact["path"])
        if not path.exists():
            errors.append(f"calibration artifact missing: {path}")
            continue
        if sha256_file(path) != artifact.get("sha256"):
            errors.append(f"calibration artifact hash mismatch: {path}")
    return errors


def _dry_run_plan(cfg: SlimderConfig, config_path: Path | None = None) -> dict:
    result = {
        "status": "dry_run",
        "teacher": {
            "load_mode": cfg.teacher.load_mode,
            "model_id_or_path": cfg.teacher.model_id_or_path,
            "revision": cfg.teacher.revision,
            "dtype": cfg.teacher.dtype,
            "device_map": cfg.teacher.device_map,
        },
        "target": cfg.compression.target.model_dump(mode="json"),
        "progressive": cfg.progressive.model_dump(mode="json"),
        "stages": [
            {
                "stage": idx + 1,
                "analyze": True,
                "compress": True,
                "distill": cfg.training.train_steps > 0,
                "eval": cfg.evaluation.perplexity.enabled,
            }
            for idx in range(cfg.progressive.stages)
        ],
        "paper_faithful": cfg.project.paper_faithful,
    }
    if config_path is not None:
        local_plan = local_dry_run_commands(config_path, cfg)
        result["local"] = {
            "commands": local_plan.commands,
            "preflight": local_plan.preflight,
            "output_dir": local_plan.output_dir,
        }
    return result


@app.callback()
def main() -> None:
    """Slimder Man CLI."""


@app.command()
def ui(test_mode: bool = False, host: str = "127.0.0.1", port: int = 7860, json_output: bool = typer.Option(False, "--json")) -> None:
    demo = create_app(test_mode=test_mode)
    if test_mode:
        _echo({"status": "started", "test_mode": True}, json_output)
        return
    demo.launch(server_name=host, server_port=port)


@app.command()
def init_config(out: Path = Path("config.yaml"), json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = tiny_default_config()
    save_config(cfg, out)
    _echo({"config": str(out)}, json_output)


@app.command()
def analyze(config: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = _load_cli_config(config)
    set_seed(cfg.project.seed)
    model = _load_model(cfg)
    arch = describe_model(model)
    tokenizer = _load_tokenizer(cfg)
    batches, cal_manifest = sample_calibration_tokens(cfg.calibration, vocab_size=arch["vocab_size"], tokenizer=tokenizer)
    if cfg.teacher.load_mode == "tiny":
        cal = collect_tiny_calibration(model, batches)
    else:
        cal = collect_calibration(model, batches, get_adapter(model))
    out_dir = Path(cfg.project.output_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = recommend(arch, cfg.compression.preset)
    write_json(out_dir / "architecture.json", arch)
    artifact_manifest = write_calibration_artifacts(out_dir, cfg, cal, cal_manifest, arch)
    write_analysis_report(out_dir / "analysis_report.md", arch, recs)
    _echo({"architecture": arch, "analysis_dir": str(out_dir), "recommendations": recs, "calibration_manifest": artifact_manifest}, json_output)


@app.command(name="recommend")
def recommend_cmd(config: Path = typer.Option(..., "--config"), preset: str = "balanced_50", json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = _load_cli_config(config)
    set_seed(cfg.project.seed)
    model = _load_model(cfg)
    recs = recommend(describe_model(model), preset)
    _echo({"preset": preset, "candidates": recs}, json_output)

@app.command()
def compress(
    config: Path | None = typer.Argument(None),
    config_option: Path | None = typer.Option(None, "--config"),
    stage: int = 1,
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    config_path = _resolve_config_path(config, config_option)
    cfg = _load_cli_config(config_path)
    set_seed(cfg.project.seed)
    out_dir = Path(cfg.project.output_dir) / "checkpoints" / f"stage_{stage}_compressed"
    teacher = _load_model(cfg)
    arch = describe_model(teacher)
    tokenizer = _load_tokenizer(cfg)
    batches, cal_manifest = sample_calibration_tokens(cfg.calibration, vocab_size=arch["vocab_size"], tokenizer=tokenizer)
    analysis_dir = Path(cfg.project.output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    recs = recommend(arch, cfg.compression.preset)
    write_json(analysis_dir / "architecture.json", arch)
    if cfg.teacher.load_mode == "tiny":
        cal = collect_tiny_calibration(teacher, batches)
        write_calibration_artifacts(analysis_dir, cfg, cal, cal_manifest, arch)
        write_analysis_report(analysis_dir / "analysis_report.md", arch, recs)
        student, manifest = compress_tiny_model(
            teacher,
            cfg,
            cal,
            out_dir,
            calibration_manifest_path=analysis_dir / "calibration_manifest.json",
            source_config_path=config_path,
        )
    else:
        adapter = get_adapter(teacher)
        cal = collect_calibration(teacher, batches, adapter)
        write_calibration_artifacts(analysis_dir, cfg, cal, cal_manifest, arch)
        write_analysis_report(analysis_dir / "analysis_report.md", arch, recs)
        student, manifest = compress_model(
            teacher,
            cfg,
            cal,
            adapter=adapter,
            output_dir=out_dir,
            tokenizer=tokenizer,
            calibration_manifest_path=analysis_dir / "calibration_manifest.json",
            source_config_path=config_path,
        )
    _echo({"checkpoint": str(out_dir), "manifest": manifest, "params": sum(p.numel() for p in student.parameters())}, json_output)


@app.command()
def distill(config: Path, stage: int = 1, resume: bool = False, json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = _load_cli_config(config)
    set_seed(cfg.project.seed)
    ckpt = Path(cfg.project.output_dir) / "checkpoints" / f"stage_{stage}_compressed"
    if cfg.teacher.load_mode == "tiny":
        teacher = _tiny_teacher(cfg)
        resume_dir = Path(cfg.project.output_dir) / "training" / "resume_model"
        if resume and resume_dir.exists():
            student = TinyMoEForCausalLM.from_pretrained(resume_dir)
        else:
            student = TinyMoEForCausalLM.from_pretrained(ckpt) if ckpt.exists() else TinyMoEForCausalLM()
        result = train_tiny_distill(teacher, student, cfg, Path(cfg.project.output_dir) / "training", resume=resume)
    else:
        if not ckpt.exists():
            raise typer.BadParameter(f"Compressed checkpoint not found: {ckpt}")
        teacher = _load_model(cfg)
        student = _load_transformers_checkpoint(cfg, ckpt)
        arch = describe_model(student)
        tokenizer = _load_tokenizer(cfg)
        batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=arch["vocab_size"], tokenizer=tokenizer)
        result = train_causal_lm_distill(teacher, student, cfg, Path(cfg.project.output_dir) / "training", batches, resume=resume)
    _echo(result, json_output)


@app.command()
def run(config: Path, dry_run: bool = typer.Option(False, "--dry-run"), json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = _load_cli_config(config)
    set_seed(cfg.project.seed)
    if dry_run:
        _echo(_dry_run_plan(cfg, config), json_output)
        return
    if cfg.teacher.load_mode != "tiny":
        if cfg.teacher.model_id_or_path != "dummy-hf-moe" and not cfg.runtime.local.allow_full_model_run:
            raise typer.BadParameter(
                "Full local run for arbitrary Transformers checkpoints requires runtime.local.allow_full_model_run=true. "
                "Use --dry-run, explicit staged analyze/compress/distill commands, or remote launch for large checkpoints."
            )
        teacher = _load_model(cfg)
        arch = describe_model(teacher)
        tokenizer = _load_tokenizer(cfg)
        batches, cal_manifest = sample_calibration_tokens(cfg.calibration, vocab_size=arch["vocab_size"], tokenizer=tokenizer)
        adapter = get_adapter(teacher)
        cal = collect_calibration(teacher, batches, adapter)
        out_dir = Path(cfg.project.output_dir)
        analysis_dir = out_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        recs = recommend(arch, cfg.compression.preset)
        write_json(analysis_dir / "architecture.json", arch)
        calibration_manifest = write_calibration_artifacts(analysis_dir, cfg, cal, cal_manifest, arch)
        write_analysis_report(analysis_dir / "analysis_report.md", arch, recs)
        ckpt_dir = out_dir / "checkpoints" / "stage_1_compressed"
        student, manifest = compress_model(
            teacher,
            cfg,
            cal,
            adapter=adapter,
            output_dir=ckpt_dir,
            tokenizer=tokenizer,
            calibration_manifest_path=analysis_dir / "calibration_manifest.json",
            source_config_path=config,
        )
        train = train_causal_lm_distill(teacher, student, cfg, out_dir / "training", batches, resume=False)
        eval_batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=arch["vocab_size"], tokenizer=tokenizer)
        ppl = causal_lm_perplexity(student, eval_batches[:8])
        result = {
            "analysis": str(analysis_dir),
            "calibration_manifest": calibration_manifest,
            "checkpoint": str(ckpt_dir),
            "manifest": manifest,
            "training": train,
            "perplexity": ppl,
            "recommendations": recs,
        }
        write_json(out_dir / "run_summary.json", result)
        _echo(result, json_output)
        return
    teacher = _tiny_teacher(cfg)
    arch = describe_model(teacher)
    batches, cal_manifest = sample_calibration_tokens(cfg.calibration, vocab_size=teacher.config.vocab_size)
    cal = collect_tiny_calibration(teacher, batches)
    out_dir = Path(cfg.project.output_dir)
    write_json(out_dir / "analysis" / "architecture.json", arch)
    write_calibration_artifacts(out_dir / "analysis", cfg, cal, cal_manifest, arch)
    write_analysis_report(out_dir / "analysis" / "analysis_report.md", arch, recommend(arch, cfg.compression.preset))
    if cfg.progressive.stages > 1 or cfg.progressive.schedule != "one_stage":
        progressive = run_tiny_progressive_stages(teacher, cfg, out_dir / "progressive", source_config_path=config)
        final_train = Path(progressive["final_training_checkpoint"])
        student = TinyMoEForCausalLM.from_pretrained(final_train)
        eval_batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)
        ppl = tiny_perplexity(student, eval_batches[:8])
        gen = student.generate(eval_batches[0][:, :2], max_new_tokens=8)
        result = {
            "analysis": str(out_dir / "analysis"),
            "progressive": progressive,
            "perplexity": ppl,
            "generated_shape": list(gen.shape),
        }
        write_json(out_dir / "run_summary.json", result)
        _echo(result, json_output)
        return
    student, manifest = compress_tiny_model(
        teacher,
        cfg,
        cal,
        out_dir / "checkpoints" / "stage_1_compressed",
        calibration_manifest_path=out_dir / "analysis" / "calibration_manifest.json",
        source_config_path=config,
    )
    train = train_tiny_distill(teacher, student, cfg, out_dir / "training", resume=False)
    eval_batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)
    ppl = tiny_perplexity(student, eval_batches[:8])
    gen = student.generate(eval_batches[0][:, :2], max_new_tokens=8)
    result = {"analysis": str(out_dir / "analysis"), "manifest": manifest, "training": train, "perplexity": ppl, "generated_shape": list(gen.shape)}
    write_json(out_dir / "run_summary.json", result)
    _echo(result, json_output)


@app.command()
def eval(
    checkpoint: Path | None = typer.Argument(None),
    checkpoint_option: Path | None = typer.Option(None, "--checkpoint"),
    tasks: str = "",
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    checkpoint = _resolve_path_arg(checkpoint, checkpoint_option, "checkpoint")
    cfg = tiny_default_config()
    set_seed(cfg.project.seed)
    model, kind = _load_checkpoint_auto(checkpoint)
    arch = describe_model(model)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=arch["vocab_size"])
    perplexity = tiny_perplexity(model, batches[:8]) if kind == "tiny" else causal_lm_perplexity(model, batches[:8])
    _echo({"checkpoint": str(checkpoint), "kind": kind, "perplexity": perplexity, "tasks": [x for x in tasks.split(",") if x]}, json_output)


@app.command()
def quantize(config: Path, checkpoint: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = load_config(config)
    set_seed(cfg.project.seed)
    if cfg.project.paper_faithful:
        raise typer.BadParameter("paper_faithful mode rejects quantization")
    out_dir = Path(cfg.project.output_dir) / "quantized"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, kind = _load_checkpoint_auto(checkpoint)
    target_bits = cfg.quantization.target_avg_bits or 8.0
    fake_manifest = fake_quantize_model(model, out_dir, target_avg_bits=target_bits, safe_serialization=cfg.student.output_format == "hf_safetensors")
    manifest = {
        "mode": cfg.quantization.mode,
        "backend": fake_manifest["backend"],
        "checkpoint_kind": kind,
        "source_checkpoint": str(checkpoint),
        "target_avg_bits": target_bits,
        "allocation": fake_manifest["allocation"],
        "validation": fake_manifest["validation"],
        "protected_modules": ["router", "gate", "norm", "embed_tokens", "lm_head", "shared"],
        "fake_quant_manifest": "fake_quant_manifest.json",
        "export_manifest": "quant_export_manifest.json",
        "artifact_hashes": {},
        "note": fake_manifest["note"],
    }
    manifest["artifact_hashes"] = {
        name: sha256_file(out_dir / name)
        for name in ("model.pt", "model.safetensors", "pytorch_model.bin", "config.json", "fake_quant_manifest.json")
        if (out_dir / name).exists()
    }
    write_json(out_dir / "quantization_manifest.json", manifest)
    write_quant_export_manifest(
        out_dir,
        manifest["backend"],
        fake_manifest,
        source_checkpoint=str(checkpoint),
    )
    _echo({"checkpoint": str(checkpoint), "out": str(out_dir), "manifest": manifest}, json_output)


@app.command()
def launch(config: Path, backend: str = "ssh", json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = _load_cli_config(config)
    if backend == "local":
        plan = local_dry_run_commands(config, cfg)
        result = {"backend": "local", "commands": plan.commands, "preflight": plan.preflight, "output_dir": plan.output_dir}
    elif backend == "ssh":
        if cfg.runtime.ssh.dry_run:
            result = {"backend": "ssh", "dry_run": True, "commands": ssh_dry_run_commands(config, cfg).commands}
        else:
            run = SSHRunner(config, cfg).launch()
            result = {
                "backend": run.backend,
                "dry_run": run.dry_run,
                "status": run.status,
                "commands": run.commands,
                "failed_command": run.failed_command,
                "results": [r.__dict__ for r in run.results],
            }
    elif backend == "skypilot":
        if cfg.runtime.skypilot.dry_run:
            result = {"backend": "skypilot", "dry_run": True, "yaml": skypilot_yaml(config, cfg)}
        else:
            run = SkyPilotRunner(config, cfg).launch()
            result = {
                "backend": run.backend,
                "dry_run": run.dry_run,
                "status": run.status,
                "task_path": run.task_path,
                "commands": run.commands,
                "failed_command": run.failed_command,
                "results": [r.__dict__ for r in run.results],
            }
    else:
        result = {"backend": backend, "status": "unsupported"}
    _echo(result, json_output)


@app.command()
def worker(
    host: str = "0.0.0.0",
    port: int = 7861,
    teacher_model: str | None = None,
    auth_token: str | None = typer.Option(None, "--auth-token", help="Bearer token required for /v1 worker endpoints."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if json_output:
        _echo({"host": host, "port": port, "auth_required": bool(auth_token or os.environ.get("SLIMDER_WORKER_TOKEN")), "endpoints": ["/v1/preflight", "/v1/jobs", "/v1/teacher_logits", "/healthz"]}, True)
        return
    import uvicorn

    from slimder_man.orchestration.worker_api import create_worker_app

    uvicorn.run(create_worker_app(teacher_model, auth_token=auth_token), host=host, port=port)


@app.command("consolidate-checkpoint")
def consolidate_checkpoint(
    checkpoint: Path | None = typer.Argument(None),
    out: Path | None = typer.Argument(None),
    checkpoint_option: Path | None = typer.Option(None, "--checkpoint"),
    out_option: Path | None = typer.Option(None, "--out"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    checkpoint = _resolve_path_arg(checkpoint, checkpoint_option, "checkpoint")
    out = _resolve_path_arg(out, out_option, "out")
    model, kind = _load_checkpoint_auto(checkpoint)
    _save_checkpoint_auto(model, out)
    sidecar_hashes = _copy_checkpoint_sidecars(checkpoint, out)
    sidecar_hashes.update(_copy_and_rewrite_calibration_references(out))
    artifact_hashes = {
        path.relative_to(out).as_posix(): sha256_file(path)
        for path in out.rglob("*")
        if path.is_file() and path.name in {"model.pt", "model.safetensors", "pytorch_model.bin", "config.json"}
    }
    artifact_hashes.update(sidecar_hashes)
    _echo({"checkpoint": str(checkpoint), "out": str(out), "kind": kind, "artifact_hashes": artifact_hashes}, json_output)


@app.command("validate-checkpoint")
def validate_checkpoint(
    checkpoint: Path | None = typer.Argument(None),
    checkpoint_option: Path | None = typer.Option(None, "--checkpoint"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    checkpoint = _resolve_path_arg(checkpoint, checkpoint_option, "checkpoint")
    model, kind = _load_checkpoint_auto(checkpoint)
    errors = validate_tiny_model(model) if kind == "tiny" else []
    errors.extend(_forward_validation_errors(model, kind))
    manifest = checkpoint / "compression_manifest.json"
    manifest_data = None
    if manifest.exists():
        try:
            manifest_data = load_manifest(manifest)
            errors.extend(_calibration_artifact_validation_errors(manifest_data))
        except Exception as exc:
            errors.append(f"compression manifest invalid: {exc}")
    _echo({"checkpoint": str(checkpoint), "kind": kind, "valid": not errors, "errors": errors, "manifest": manifest_data}, json_output)


if __name__ == "__main__":
    app()

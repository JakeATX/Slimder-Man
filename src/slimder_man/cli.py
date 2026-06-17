from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich import print

from slimder_man.adapters.registry import get_adapter
from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.analyze.architecture import describe_model
from slimder_man.analyze.recommender import recommend
from slimder_man.analyze.reports import write_analysis_report
from slimder_man.calibration.artifacts import write_calibration_artifacts
from slimder_man.calibration.collectors import collect_calibration, collect_tiny_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import compress_model, compress_tiny_model
from slimder_man.compression.manifests import load_manifest
from slimder_man.compression.validate import validate_tiny_model
from slimder_man.config.defaults import tiny_default_config
from slimder_man.config.schema import SlimderConfig, load_config, save_config
from slimder_man.distill.train_loop import train_causal_lm_distill, train_tiny_distill
from slimder_man.distill.stage_runner import run_tiny_progressive_stages
from slimder_man.eval.perplexity import causal_lm_perplexity, tiny_perplexity
from slimder_man.orchestration.skypilot import skypilot_yaml
from slimder_man.orchestration.ssh import ssh_dry_run_commands
from slimder_man.quant.fake_backend import fake_quantize_tiny_model
from slimder_man.ui.app import create_app
from slimder_man.utils.hashing import sha256_file
from slimder_man.utils.json import write_json
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


def _dry_run_plan(cfg: SlimderConfig) -> dict:
    return {
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
    cfg = _load_cli_config(_resolve_config_path(config, config_option))
    set_seed(cfg.project.seed)
    out_dir = Path(cfg.project.output_dir) / "checkpoints" / f"stage_{stage}_compressed"
    teacher = _load_model(cfg)
    arch = describe_model(teacher)
    tokenizer = _load_tokenizer(cfg)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=arch["vocab_size"], tokenizer=tokenizer)
    if cfg.teacher.load_mode == "tiny":
        cal = collect_tiny_calibration(teacher, batches)
        student, manifest = compress_tiny_model(teacher, cfg, cal, out_dir)
    else:
        adapter = get_adapter(teacher)
        cal = collect_calibration(teacher, batches, adapter)
        student, manifest = compress_model(teacher, cfg, cal, adapter=adapter, output_dir=out_dir, tokenizer=tokenizer)
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
        _echo(_dry_run_plan(cfg), json_output)
        return
    if cfg.teacher.load_mode != "tiny":
        if cfg.teacher.model_id_or_path != "dummy-hf-moe":
            raise typer.BadParameter(
                "Full local run is currently enabled only for the bundled dummy-hf-moe smoke fixture; "
                "use --dry-run plus explicit analyze/compress/distill or launch remote orchestration for real checkpoints."
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
        student, manifest = compress_model(teacher, cfg, cal, adapter=adapter, output_dir=ckpt_dir, tokenizer=tokenizer)
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
        progressive = run_tiny_progressive_stages(teacher, cfg, out_dir / "progressive")
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
    student, manifest = compress_tiny_model(teacher, cfg, cal, out_dir / "checkpoints" / "stage_1_compressed")
    train = train_tiny_distill(teacher, student, cfg, out_dir / "training", resume=False)
    eval_batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)
    ppl = tiny_perplexity(student, eval_batches[:8])
    gen = student.generate(eval_batches[0][:, :2], max_new_tokens=8)
    result = {"analysis": str(out_dir / "analysis"), "manifest": manifest, "training": train, "perplexity": ppl, "generated_shape": list(gen.shape)}
    write_json(out_dir / "run_summary.json", result)
    _echo(result, json_output)


@app.command()
def eval(checkpoint: Path, tasks: str = "", json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = tiny_default_config()
    set_seed(cfg.project.seed)
    model = TinyMoEForCausalLM.from_pretrained(checkpoint)
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=model.config.vocab_size)
    _echo({"perplexity": tiny_perplexity(model, batches[:8]), "tasks": [x for x in tasks.split(",") if x]}, json_output)


@app.command()
def quantize(config: Path, checkpoint: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = load_config(config)
    set_seed(cfg.project.seed)
    if cfg.project.paper_faithful:
        raise typer.BadParameter("paper_faithful mode rejects quantization")
    if cfg.teacher.load_mode != "tiny":
        raise typer.BadParameter("Only the fake tiny quantization backend is implemented; HF AWQ/GPTQ/SmoothQuant/bnb adapters remain explicit future backends.")
    out_dir = Path(cfg.project.output_dir) / "quantized"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = TinyMoEForCausalLM.from_pretrained(checkpoint)
    target_bits = cfg.quantization.target_avg_bits or 8.0
    fake_manifest = fake_quantize_tiny_model(model, out_dir, target_avg_bits=target_bits)
    artifact_hashes = {
        name: sha256_file(out_dir / name)
        for name in ("model.pt", "config.json", "fake_quant_manifest.json")
        if (out_dir / name).exists()
    }
    manifest = {
        "mode": cfg.quantization.mode,
        "backend": fake_manifest["backend"],
        "source_checkpoint": str(checkpoint),
        "target_avg_bits": target_bits,
        "allocation": fake_manifest["allocation"],
        "validation": fake_manifest["validation"],
        "protected_modules": ["router", "norm", "embed_tokens", "lm_head", "shared"],
        "fake_quant_manifest": "fake_quant_manifest.json",
        "artifact_hashes": artifact_hashes,
        "note": fake_manifest["note"],
    }
    write_json(out_dir / "quantization_manifest.json", manifest)
    _echo({"checkpoint": str(checkpoint), "out": str(out_dir), "manifest": manifest}, json_output)


@app.command()
def launch(config: Path, backend: str = "ssh", json_output: bool = typer.Option(False, "--json")) -> None:
    cfg = load_config(config)
    if backend == "ssh":
        result = {"backend": "ssh", "commands": ssh_dry_run_commands(cfg).commands}
    elif backend == "skypilot":
        result = {"backend": "skypilot", "yaml": skypilot_yaml(cfg)}
    else:
        result = {"backend": backend, "status": "local"}
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
def consolidate_checkpoint(checkpoint: Path, out: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    model = TinyMoEForCausalLM.from_pretrained(checkpoint)
    model.save_pretrained(out)
    _echo({"checkpoint": str(checkpoint), "out": str(out)}, json_output)


@app.command("validate-checkpoint")
def validate_checkpoint(checkpoint: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    model = TinyMoEForCausalLM.from_pretrained(checkpoint)
    errors = validate_tiny_model(model)
    manifest = checkpoint / "compression_manifest.json"
    if manifest.exists():
        load_manifest(manifest)
    _echo({"checkpoint": str(checkpoint), "valid": not errors, "errors": errors}, json_output)


if __name__ == "__main__":
    app()

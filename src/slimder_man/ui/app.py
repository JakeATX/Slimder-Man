from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from slimder_man.config.schema import SlimderConfig
from slimder_man.config.defaults import tiny_default_config


def build_config_yaml(
    project_name: str = "tiny_moe_cpu_smoke",
    paper_faithful: bool = True,
    quantization: bool = False,
    teacher_model_id_or_path: str = "tiny",
    teacher_load_mode: str = "tiny",
    teacher_dtype: str = "float32",
    teacher_revision: str = "",
    trust_remote_code: bool = True,
    dataset_type: str = "synthetic",
    dataset_name: str = "",
    dataset_path: str = "",
    dataset_split: str = "train",
    text_field: str = "text",
    sample_count: int = 16,
    sequence_length: int = 16,
    token_budget: int = 1024,
    train_steps: int = 5,
    runtime_backend: str = "local",
    local_num_gpus: str = "auto",
    ssh_host: str = "",
    ssh_user: str = "",
    ssh_port: int = 22,
    ssh_dry_run: bool = True,
    skypilot_cluster_name: str = "slimder",
    skypilot_accelerators: str = "H100:8",
    skypilot_cloud: str = "auto",
    tracking_backend: str = "tensorboard",
    output_dir: str | None = None,
    compression_preset: str | None = None,
    local_allow_full_model_run: bool = False,
    skypilot_region: str = "",
    skypilot_image_id: str = "",
    skypilot_disk_size_gb: int = 512,
    skypilot_autostop_minutes: int = 60,
    skypilot_dry_run: bool = True,
    worker_api_url: str = "",
    worker_auth_token_env: str = "SLIMDER_WORKER_TOKEN",
    worker_timeout_seconds: float = 60.0,
    kd_teacher_mode: str = "online_full_logits",
    offline_full_logits_cache_path: str = "",
    allow_smoke_trainer: bool = False,
) -> str:
    import yaml

    cfg = tiny_default_config()
    cfg.project.name = project_name
    cfg.project.output_dir = output_dir or f"runs/{project_name}"
    cfg.project.paper_faithful = paper_faithful
    cfg.quantization.enabled = bool(quantization) and not paper_faithful
    cfg.teacher.model_id_or_path = teacher_model_id_or_path
    cfg.teacher.load_mode = teacher_load_mode
    cfg.teacher.dtype = teacher_dtype
    cfg.teacher.revision = teacher_revision or None
    cfg.teacher.trust_remote_code = trust_remote_code
    cfg.calibration.dataset.type = dataset_type
    cfg.calibration.dataset.name = dataset_name or None
    cfg.calibration.dataset.path = dataset_path or None
    cfg.calibration.dataset.split = dataset_split
    cfg.calibration.dataset.text_field = text_field
    cfg.calibration.sample_count = int(sample_count)
    cfg.calibration.sequence_length = int(sequence_length)
    cfg.training.token_budget = int(token_budget)
    cfg.training.sequence_length = int(sequence_length)
    cfg.training.train_steps = int(train_steps)
    cfg.training.allow_smoke_trainer = bool(allow_smoke_trainer)
    if compression_preset:
        cfg.compression.preset = compression_preset
    cfg.runtime.backend = runtime_backend
    cfg.runtime.local.num_gpus = "auto" if local_num_gpus == "auto" else int(local_num_gpus)
    cfg.runtime.local.allow_full_model_run = bool(local_allow_full_model_run)
    cfg.runtime.ssh.host = ssh_host or None
    cfg.runtime.ssh.user = ssh_user or None
    cfg.runtime.ssh.port = int(ssh_port)
    cfg.runtime.ssh.dry_run = ssh_dry_run
    cfg.runtime.skypilot.cluster_name = skypilot_cluster_name
    cfg.runtime.skypilot.accelerators = skypilot_accelerators
    cfg.runtime.skypilot.cloud = skypilot_cloud
    cfg.runtime.skypilot.region = skypilot_region or None
    cfg.runtime.skypilot.image_id = skypilot_image_id or None
    cfg.runtime.skypilot.disk_size_gb = int(skypilot_disk_size_gb)
    cfg.runtime.skypilot.autostop_minutes = int(skypilot_autostop_minutes)
    cfg.runtime.skypilot.dry_run = bool(skypilot_dry_run)
    cfg.runtime.worker.api_url = worker_api_url or None
    cfg.runtime.worker.auth_token_env = worker_auth_token_env or None
    cfg.runtime.worker.timeout_seconds = float(worker_timeout_seconds)
    cfg.kd.teacher_mode = kd_teacher_mode
    cfg.kd.offline_full_logits_cache_path = offline_full_logits_cache_path or None
    cfg.runtime.tracking.backend = tracking_backend
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)


def _config_from_yaml(yaml_text: str) -> SlimderConfig:
    import yaml

    return SlimderConfig.model_validate(yaml.safe_load(yaml_text))


def _json_or_error(proc: subprocess.CompletedProcess[str]) -> str:
    return proc.stdout if proc.returncode == 0 else proc.stderr


def apply_candidate_to_yaml(yaml_text: str, preset: str = "balanced_50", candidate_id: str = "") -> str:
    if preset == "all":
        raise ValueError("Choose one preset before applying a candidate; preset=all is only for comparison.")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = _config_from_yaml(yaml_text)
        chosen = candidate_id or f"{preset}_1"
        argv = [
            sys.executable,
            "-m",
            "slimder_man.cli",
            "recommend",
            "--config",
            str(path),
            "--preset",
            preset,
            "--candidate-id",
            chosen,
            "--write-config",
            str(path),
            "--json",
        ]
        if cfg.teacher.load_mode == "transformers" and cfg.teacher.model_id_or_path != "dummy-hf-moe":
            argv.append("--config-only")
        proc = subprocess.run(argv, text=True, capture_output=True)
        if proc.returncode != 0:
            raise ValueError(proc.stderr or proc.stdout)
        return path.read_text(encoding="utf-8")


def run_cli_with_yaml(
    yaml_text: str,
    command: str,
    preset: str = "balanced_50",
    stage: int = 1,
    backend: str = "local",
    checkpoint: str | None = None,
) -> str:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "config.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = _config_from_yaml(yaml_text)
        if command == "recommend":
            argv = [sys.executable, "-m", "slimder_man.cli", "recommend", "--config", str(path), "--preset", preset, "--json"]
        elif command == "compress":
            argv = [sys.executable, "-m", "slimder_man.cli", "compress", "--config", str(path), "--stage", str(stage), "--json"]
        elif command == "distill":
            argv = [sys.executable, "-m", "slimder_man.cli", "distill", str(path), "--stage", str(stage), "--json"]
        elif command == "eval":
            ckpt = checkpoint or str(Path(cfg.project.output_dir) / "training" / "final")
            argv = [sys.executable, "-m", "slimder_man.cli", "eval", "--checkpoint", ckpt, "--json"]
        elif command == "launch":
            argv = [sys.executable, "-m", "slimder_man.cli", "launch", str(path), "--backend", backend, "--json"]
        else:
            argv = [sys.executable, "-m", "slimder_man.cli", command, str(path), "--json"]
        proc = subprocess.run(argv, text=True, capture_output=True)
        return _json_or_error(proc)


def run_ui_command(
    command: str,
    *config_args,
    preset: str = "balanced_50",
    stage: int = 1,
    backend: str = "local",
    checkpoint: str | None = None,
    compression_preset: str | None = None,
) -> str:
    return run_cli_with_yaml(
        build_config_yaml(*config_args, compression_preset=compression_preset),
        command,
        preset=preset,
        stage=stage,
        backend=backend,
        checkpoint=checkpoint,
    )


def run_ui_yaml_command(
    yaml_text: str,
    command: str,
    preset: str = "balanced_50",
    stage: int = 1,
    backend: str = "local",
    checkpoint: str | None = None,
) -> str:
    return run_cli_with_yaml(
        yaml_text,
        command,
        preset=preset,
        stage=stage,
        backend=backend,
        checkpoint=checkpoint,
    )


def artifact_index(output_dir: str) -> str:
    out = Path(output_dir or "runs/tiny_moe_cpu_smoke")
    if not out.exists():
        return f"No artifacts found under {out}"
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name in {
            "analysis_report.md",
            "compression_manifest.json",
            "training_report.md",
            "run_summary.json",
            "quantization_manifest.json",
            "fake_quant_manifest.json",
            "calibration_manifest.json",
            "trainer_state.json",
        }:
            rows.append(str(path.resolve()))
    return "\n".join(rows) if rows else f"No known Slimder artifacts found under {out}"


def log_tail(output_dir: str, max_lines: int = 80) -> str:
    out = Path(output_dir or "runs/tiny_moe_cpu_smoke")
    candidates = [
        out / "training" / "training_report.md",
        out / "progressive" / "stage_1" / "training" / "training_report.md",
        out / "progressive" / "stage_2" / "training" / "training_report.md",
        out / "run_summary.json",
    ]
    lines: list[str] = []
    for path in candidates:
        if path.exists():
            lines.append(f"# {path.resolve()}")
            lines.extend(path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:])
    return "\n".join(lines) if lines else f"No logs found under {out}"


def config_warnings(yaml_text: str) -> str:
    cfg = _config_from_yaml(yaml_text)
    warnings: list[str] = []
    if cfg.compression.preset in {"aggressive_80", "extreme_90"}:
        warnings.append(f"{cfg.compression.preset} is high-risk and should be validated with staged evals.")
    if not cfg.project.paper_faithful:
        warnings.append("Non-paper-faithful mode enables augmented behavior; manifests should be reviewed for contamination.")
    if cfg.teacher.load_mode == "transformers" and cfg.runtime.backend == "local" and not cfg.runtime.local.allow_full_model_run:
        warnings.append("Arbitrary Transformers local run requires runtime.local.allow_full_model_run=true; use launch/dry-run for large checkpoints.")
    if cfg.teacher.load_mode == "transformers" and cfg.teacher.model_id_or_path != "dummy-hf-moe" and not cfg.training.allow_smoke_trainer:
        warnings.append("Single-process Transformers distillation requires training.allow_smoke_trainer=true and should be limited to explicit small-model smoke runs.")
    return "\n".join(warnings) if warnings else "No warnings."


def paper_faithful_quant_state(paper_faithful: bool) -> dict:
    if paper_faithful:
        return {"value": False, "interactive": False}
    return {"interactive": True}


def create_app(test_mode: bool = False):
    try:
        import gradio as gr
    except Exception:
        if test_mode:
            return {"test_mode": True, "yaml": build_config_yaml()}
        raise
    with gr.Blocks(title="Slimder Man") as demo:
        gr.Markdown("# Slimder Man")
        with gr.Tabs():
            with gr.Tab("Project"):
                project = gr.Textbox(value="tiny_moe_cpu_smoke", label="Project name")
                output_dir = gr.Textbox(value="runs/tiny_moe_cpu_smoke", label="Output directory")
                faithful = gr.Checkbox(value=True, label="Paper faithful")
                quant = gr.Checkbox(value=False, label="Augmented quantization", interactive=False)
            with gr.Tab("Teacher & Dataset"):
                teacher_model = gr.Textbox(value="tiny", label="Model ID or path")
                teacher_load = gr.Dropdown(["tiny", "transformers"], value="tiny", label="Load mode")
                teacher_dtype = gr.Dropdown(["float32", "bfloat16", "float16"], value="float32", label="Teacher dtype")
                teacher_revision = gr.Textbox(value="", label="Revision")
                trust_remote = gr.Checkbox(value=True, label="Trust remote code")
                dataset_type = gr.Dropdown(["synthetic", "hf_dataset", "jsonl", "parquet", "text", "tokenized"], value="synthetic", label="Dataset type")
                dataset_name = gr.Textbox(value="", label="Dataset name")
                dataset_path = gr.Textbox(value="", label="Dataset path")
                dataset_split = gr.Textbox(value="train", label="Split")
                text_field = gr.Textbox(value="text", label="Text field")
            with gr.Tab("Runtime"):
                sample_count = gr.Number(value=16, precision=0, label="Calibration samples")
                sequence_length = gr.Number(value=16, precision=0, label="Sequence length")
                token_budget = gr.Number(value=1024, precision=0, label="Token budget")
                train_steps = gr.Number(value=5, precision=0, label="Train steps")
                allow_smoke_trainer = gr.Checkbox(value=False, label="Allow smoke trainer")
                kd_teacher_mode = gr.Dropdown(["online_full_logits", "offline_full_logits_cache", "remote_worker_full_logits"], value="online_full_logits", label="KD teacher mode")
                offline_cache = gr.Textbox(value="", label="Offline full-logit cache")
                runtime_backend = gr.Dropdown(["local", "ssh", "skypilot", "worker"], value="local", label="Runtime backend")
                local_gpus = gr.Textbox(value="auto", label="Local GPUs")
                local_allow_full_model = gr.Checkbox(value=False, label="Allow local full-model run")
                ssh_host = gr.Textbox(value="", label="SSH host")
                ssh_user = gr.Textbox(value="", label="SSH user")
                ssh_port = gr.Number(value=22, precision=0, label="SSH port")
                ssh_dry = gr.Checkbox(value=True, label="SSH dry run")
                sky_cluster = gr.Textbox(value="slimder", label="SkyPilot cluster")
                sky_accelerators = gr.Textbox(value="H100:8", label="SkyPilot accelerators")
                sky_cloud = gr.Textbox(value="auto", label="SkyPilot cloud")
                sky_region = gr.Textbox(value="", label="SkyPilot region")
                sky_image = gr.Textbox(value="", label="SkyPilot image")
                sky_disk = gr.Number(value=512, precision=0, label="SkyPilot disk GB")
                sky_autostop = gr.Number(value=60, precision=0, label="SkyPilot autostop minutes")
                sky_dry = gr.Checkbox(value=True, label="SkyPilot dry run")
                worker_url = gr.Textbox(value="", label="Worker API URL")
                worker_token_env = gr.Textbox(value="SLIMDER_WORKER_TOKEN", label="Worker token env")
                worker_timeout = gr.Number(value=60.0, label="Worker timeout seconds")
                tracking = gr.Dropdown(["tensorboard", "wandb", "mlflow", "none"], value="tensorboard", label="Tracking")
            with gr.Tab("Analyze"):
                analyze_btn = gr.Button("Run Analyze")
                analyze_out = gr.Code(language="json", label="Analyze Output")
            with gr.Tab("Recommendations"):
                preset = gr.Dropdown(["conservative_20", "balanced_50", "slimqwen_anchor", "aggressive_80", "extreme_90", "all"], value="balanced_50", label="Preset")
                with gr.Row():
                    conservative_btn = gr.Button("20 percent")
                    balanced_btn = gr.Button("50 percent")
                    anchor_btn = gr.Button("SlimQwen")
                    aggressive_btn = gr.Button("80 percent")
                    extreme_btn = gr.Button("90 percent")
                recommend_btn = gr.Button("Generate Recommendations")
                apply_candidate_btn = gr.Button("Apply Top Candidate")
                recommend_out = gr.Code(language="json", label="Recommendation Output")
            with gr.Tab("Compression"):
                stage = gr.Number(value=1, precision=0, label="Stage")
                compress_btn = gr.Button("Run Compression")
                compress_out = gr.Code(language="json", label="Compression Output")
            with gr.Tab("Training / Distillation"):
                distill_btn = gr.Button("Run Distillation")
                distill_out = gr.Code(language="json", label="Distillation Output")
                logs_btn = gr.Button("Refresh Logs")
                logs_out = gr.Textbox(lines=12, label="Logs")
            with gr.Tab("Evaluation"):
                eval_checkpoint = gr.Textbox(value="runs/tiny_moe_cpu_smoke/training/final", label="Checkpoint")
                eval_btn = gr.Button("Run Evaluation")
                eval_out = gr.Code(language="json", label="Evaluation Output")
            with gr.Tab("Artifacts"):
                launch_backend = gr.Dropdown(["local", "ssh", "skypilot", "worker"], value="local", label="Launch backend")
                launch_btn = gr.Button("Launch / Dry Run")
                launch_out = gr.Code(language="json", label="Launch Output")
                artifacts_btn = gr.Button("Refresh Artifacts")
                artifacts_out = gr.Textbox(lines=12, label="Artifacts")
                warnings_out = gr.Textbox(value="No warnings.", lines=4, label="Warnings")
            with gr.Tab("Config"):
                output = gr.Code(value=build_config_yaml(), label="Generated YAML", language="yaml")
                btn = gr.Button("Generate Config")
                run_btn = gr.Button("Run Tiny Pipeline")
                run_out = gr.Code(language="json", label="Run Output")
        def _build_from_ui(project_name, paper_faithful, quantization, teacher_model_id_or_path, teacher_load_mode, teacher_dtype, teacher_revision, trust_remote_code, dataset_type, dataset_name, dataset_path, dataset_split, text_field, sample_count, sequence_length, token_budget, train_steps, runtime_backend, local_num_gpus, local_allow_full_model_run, ssh_host, ssh_user, ssh_port, ssh_dry_run, skypilot_cluster_name, skypilot_accelerators, skypilot_cloud, skypilot_region, skypilot_image_id, skypilot_disk_size_gb, skypilot_autostop_minutes, skypilot_dry_run, worker_api_url, worker_auth_token_env, worker_timeout_seconds, kd_teacher_mode, offline_full_logits_cache_path, allow_smoke_trainer_value, tracking_backend, output_dir_value, compression_preset):
            return build_config_yaml(
                project_name=project_name,
                paper_faithful=paper_faithful,
                quantization=quantization,
                teacher_model_id_or_path=teacher_model_id_or_path,
                teacher_load_mode=teacher_load_mode,
                teacher_dtype=teacher_dtype,
                teacher_revision=teacher_revision,
                trust_remote_code=trust_remote_code,
                dataset_type=dataset_type,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                dataset_split=dataset_split,
                text_field=text_field,
                sample_count=sample_count,
                sequence_length=sequence_length,
                token_budget=token_budget,
                train_steps=train_steps,
                runtime_backend=runtime_backend,
                local_num_gpus=local_num_gpus,
                ssh_host=ssh_host,
                ssh_user=ssh_user,
                ssh_port=ssh_port,
                ssh_dry_run=ssh_dry_run,
                skypilot_cluster_name=skypilot_cluster_name,
                skypilot_accelerators=skypilot_accelerators,
                skypilot_cloud=skypilot_cloud,
                tracking_backend=tracking_backend,
                output_dir=output_dir_value,
                compression_preset=compression_preset,
                local_allow_full_model_run=local_allow_full_model_run,
                skypilot_region=skypilot_region,
                skypilot_image_id=skypilot_image_id,
                skypilot_disk_size_gb=skypilot_disk_size_gb,
                skypilot_autostop_minutes=skypilot_autostop_minutes,
                skypilot_dry_run=skypilot_dry_run,
                worker_api_url=worker_api_url,
                worker_auth_token_env=worker_auth_token_env,
                worker_timeout_seconds=worker_timeout_seconds,
                kd_teacher_mode=kd_teacher_mode,
                offline_full_logits_cache_path=offline_full_logits_cache_path,
                allow_smoke_trainer=allow_smoke_trainer_value,
            )

        def _run_from_yaml(yaml_text, command, preset_value="balanced_50", stage_value=1, backend_value="local", checkpoint_value=None):
            return run_ui_yaml_command(
                yaml_text,
                command,
                preset=preset_value,
                stage=int(stage_value),
                backend=backend_value,
                checkpoint=checkpoint_value,
            )

        config_inputs = [
            project,
            faithful,
            quant,
            teacher_model,
            teacher_load,
            teacher_dtype,
            teacher_revision,
            trust_remote,
            dataset_type,
            dataset_name,
            dataset_path,
            dataset_split,
            text_field,
            sample_count,
            sequence_length,
            token_budget,
            train_steps,
            runtime_backend,
            local_gpus,
            local_allow_full_model,
            ssh_host,
            ssh_user,
            ssh_port,
            ssh_dry,
            sky_cluster,
            sky_accelerators,
            sky_cloud,
            sky_region,
            sky_image,
            sky_disk,
            sky_autostop,
            sky_dry,
            worker_url,
            worker_token_env,
            worker_timeout,
            kd_teacher_mode,
            offline_cache,
            allow_smoke_trainer,
            tracking,
            output_dir,
        ]
        btn.click(_build_from_ui, inputs=[*config_inputs, preset], outputs=[output]).then(config_warnings, inputs=[output], outputs=[warnings_out])
        faithful.change(lambda value: gr.update(**paper_faithful_quant_state(value)), inputs=[faithful], outputs=[quant])
        for value, button in [
            ("conservative_20", conservative_btn),
            ("balanced_50", balanced_btn),
            ("slimqwen_anchor", anchor_btn),
            ("aggressive_80", aggressive_btn),
            ("extreme_90", extreme_btn),
        ]:
            button.click(lambda selected=value: selected, outputs=[preset])
        analyze_btn.click(lambda yaml_text, selected: _run_from_yaml(yaml_text, "analyze", preset_value=selected), inputs=[output, preset], outputs=[analyze_out])
        recommend_btn.click(
            lambda yaml_text, selected: _run_from_yaml(yaml_text, "recommend", preset_value=selected),
            inputs=[output, preset],
            outputs=[recommend_out],
        )
        apply_candidate_btn.click(
            apply_candidate_to_yaml,
            inputs=[output, preset],
            outputs=[output],
        ).then(config_warnings, inputs=[output], outputs=[warnings_out])
        compress_btn.click(
            lambda yaml_text, stage_value, selected: _run_from_yaml(yaml_text, "compress", preset_value=selected, stage_value=stage_value),
            inputs=[output, stage, preset],
            outputs=[compress_out],
        )
        distill_btn.click(
            lambda yaml_text, stage_value, selected: _run_from_yaml(yaml_text, "distill", preset_value=selected, stage_value=stage_value),
            inputs=[output, stage, preset],
            outputs=[distill_out],
        )
        eval_btn.click(
            lambda yaml_text, checkpoint_value, selected: _run_from_yaml(yaml_text, "eval", preset_value=selected, checkpoint_value=checkpoint_value),
            inputs=[output, eval_checkpoint, preset],
            outputs=[eval_out],
        )
        launch_btn.click(
            lambda yaml_text, backend_value, selected: _run_from_yaml(yaml_text, "launch", preset_value=selected, backend_value=backend_value),
            inputs=[output, launch_backend, preset],
            outputs=[launch_out],
        )
        run_btn.click(lambda yaml_text, selected: _run_from_yaml(yaml_text, "run", preset_value=selected), inputs=[output, preset], outputs=[run_out])
        logs_btn.click(log_tail, inputs=[output_dir], outputs=[logs_out])
        artifacts_btn.click(artifact_index, inputs=[output_dir], outputs=[artifacts_out])
    return demo

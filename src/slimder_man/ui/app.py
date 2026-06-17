from __future__ import annotations

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
) -> str:
    import yaml

    cfg = tiny_default_config()
    cfg.project.name = project_name
    cfg.project.output_dir = f"runs/{project_name}"
    cfg.project.paper_faithful = paper_faithful
    cfg.quantization.enabled = quantization
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
    cfg.runtime.backend = runtime_backend
    cfg.runtime.local.num_gpus = "auto" if local_num_gpus == "auto" else int(local_num_gpus)
    cfg.runtime.ssh.host = ssh_host or None
    cfg.runtime.ssh.user = ssh_user or None
    cfg.runtime.ssh.port = int(ssh_port)
    cfg.runtime.ssh.dry_run = ssh_dry_run
    cfg.runtime.skypilot.cluster_name = skypilot_cluster_name
    cfg.runtime.skypilot.accelerators = skypilot_accelerators
    cfg.runtime.skypilot.cloud = skypilot_cloud
    cfg.runtime.tracking.backend = tracking_backend
    return yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False)


def create_app(test_mode: bool = False):
    try:
        import gradio as gr
    except Exception:
        if test_mode:
            return {"test_mode": True, "yaml": build_config_yaml()}
        raise
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    def _run_generated(yaml_text: str, command: str) -> str:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "config.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            proc = subprocess.run([sys.executable, "-m", "slimder_man.cli", command, str(path), "--json"], text=True, capture_output=True)
            return proc.stdout if proc.returncode == 0 else proc.stderr

    with gr.Blocks(title="Slimder Man") as demo:
        gr.Markdown("# Slimder Man")
        with gr.Tabs():
            with gr.Tab("Project"):
                project = gr.Textbox(value="tiny_moe_cpu_smoke", label="Project name")
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
                runtime_backend = gr.Dropdown(["local", "ssh", "skypilot", "worker"], value="local", label="Runtime backend")
                local_gpus = gr.Textbox(value="auto", label="Local GPUs")
                ssh_host = gr.Textbox(value="", label="SSH host")
                ssh_user = gr.Textbox(value="", label="SSH user")
                ssh_port = gr.Number(value=22, precision=0, label="SSH port")
                ssh_dry = gr.Checkbox(value=True, label="SSH dry run")
                sky_cluster = gr.Textbox(value="slimder", label="SkyPilot cluster")
                sky_accelerators = gr.Textbox(value="H100:8", label="SkyPilot accelerators")
                sky_cloud = gr.Textbox(value="auto", label="SkyPilot cloud")
                tracking = gr.Dropdown(["tensorboard", "wandb", "mlflow", "none"], value="tensorboard", label="Tracking")
            with gr.Tab("Analyze"):
                analyze_btn = gr.Button("Run Analyze")
                analyze_out = gr.Code(language="json", label="Analyze Output")
            with gr.Tab("Recommendations"):
                gr.Markdown("Preset recommendations are generated by the CLI and run pipeline.")
            with gr.Tab("Compression"):
                gr.Markdown("Compression runs as part of `slimder run` for the tiny smoke workflow.")
            with gr.Tab("Training / Distillation"):
                gr.Markdown("Training logs are written to `training_report.md`.")
            with gr.Tab("Evaluation"):
                gr.Markdown("Perplexity summaries are written to `run_summary.json`.")
            with gr.Tab("Artifacts"):
                gr.Markdown("Checkpoints, manifests, reports, and hashes are written under `project.output_dir`.")
            output = gr.Code(value=build_config_yaml(), label="Generated YAML", language="yaml")
            btn = gr.Button("Generate Config")
            run_btn = gr.Button("Run Tiny Pipeline")
            run_out = gr.Code(language="json", label="Run Output")
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
            ssh_host,
            ssh_user,
            ssh_port,
            ssh_dry,
            sky_cluster,
            sky_accelerators,
            sky_cloud,
            tracking,
        ]
        btn.click(build_config_yaml, inputs=config_inputs, outputs=[output])
        analyze_btn.click(lambda y: _run_generated(y, "analyze"), inputs=[output], outputs=[analyze_out])
        run_btn.click(lambda y: _run_generated(y, "run"), inputs=[output], outputs=[run_out])
    return demo

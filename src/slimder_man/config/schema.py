from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictBaseModel):
    name: str = "tiny_moe_cpu_smoke"
    output_dir: str = "runs/tiny_moe_cpu_smoke"
    seed: int = 1337
    paper_faithful: bool = True


class TeacherConfig(StrictBaseModel):
    model_id_or_path: str = "tiny"
    revision: str | None = None
    dtype: str = "float32"
    trust_remote_code: bool = True
    load_mode: Literal["tiny", "transformers"] = "tiny"
    tensor_parallel: int | str | None = None
    device_map: str | None = None


class StudentConfig(StrictBaseModel):
    init_from: Literal["teacher_pruned", "scratch"] = "teacher_pruned"
    output_format: Literal["hf_safetensors", "torch"] = "torch"
    tie_embeddings: bool | Literal["auto"] = "auto"


class DatasetConfig(StrictBaseModel):
    type: Literal["synthetic", "hf_dataset", "jsonl", "parquet", "text", "tokenized"] = "synthetic"
    name: str | None = None
    path: str | None = None
    split: str = "train"
    text_field: str = "text"


class CalibrationConfig(StrictBaseModel):
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    sample_count: int = 16
    sequence_length: int = 16
    packing: bool = True
    seed: int = 1337
    expert_balanced_sampling: bool = False


class CompressionTarget(StrictBaseModel):
    total_param_reduction: float | None = None
    active_param_reduction: float | None = None
    hidden_size: int = 12
    depth_remove_fraction: float | None = None
    remove_last_n_layers: int = 1
    routed_experts: int = 4
    routed_top_k: int = 2
    shared_experts: Literal["keep", "prune"] = "keep"


class DepthConfig(StrictBaseModel):
    method: Literal["last_layers", "activation_similarity"] = "last_layers"


class WidthConfig(StrictBaseModel):
    method: Literal["rmsnorm_mean_abs", "magnitude"] = "rmsnorm_mean_abs"
    hidden_size_multiple: int = 4


class ExpertsConfig(StrictBaseModel):
    method: Literal["partial_preservation_merge", "prune"] = "partial_preservation_merge"
    importance_metric: Literal["frequency", "soft_logits", "reap"] = "soft_logits"
    similarity_metric: Literal["router_logits", "router_weights", "expert_outputs"] = "router_weights"
    base_selection: Literal["next_highest_importance", "diverse_kcenter"] = "next_highest_importance"
    router_row_strategy: Literal["base", "weighted_average", "reinit_then_train"] = "base"
    keep_shared_experts: bool = True


class CompressionConfig(StrictBaseModel):
    preset: str = "balanced_50"
    target: CompressionTarget = Field(default_factory=CompressionTarget)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    width: WidthConfig = Field(default_factory=WidthConfig)
    experts: ExpertsConfig = Field(default_factory=ExpertsConfig)


class ProgressiveConfig(StrictBaseModel):
    schedule: Literal["one_stage", "depth_first", "width_first", "joint"] = "one_stage"
    stages: int = 1
    token_split: list[float] = Field(default_factory=lambda: [1.0])

    @model_validator(mode="after")
    def validate_split(self) -> "ProgressiveConfig":
        if self.stages not in {1, 2}:
            raise ValueError("progressive.stages currently supports only 1 or 2 stages")
        if self.schedule == "one_stage" and self.stages != 1:
            raise ValueError("progressive.schedule=one_stage requires stages=1")
        if self.stages == 2 and self.token_split == [1.0]:
            self.token_split = [0.1, 0.9]
        if self.stages != len(self.token_split):
            raise ValueError("progressive.stages must match token_split length")
        if abs(sum(self.token_split) - 1.0) > 1e-6:
            raise ValueError("progressive.token_split must sum to 1.0")
        return self


class MoeAuxLossConfig(StrictBaseModel):
    global_batch_load_balancing: bool = True
    weight: float | Literal["adapter_default"] = "adapter_default"


class TrainingConfig(StrictBaseModel):
    token_budget: int = 1024
    global_batch_size: int | Literal["auto"] = 4
    micro_batch_size: int | Literal["auto"] = 2
    sequence_length: int = 16
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = 4.0e-4
    min_learning_rate: float = 3.0e-5
    lr_schedule: Literal["cosine_decay"] = "cosine_decay"
    warmup_steps: int = 2000
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    train_steps: int = 5
    save_every_steps: int = 1000
    moe_aux_loss: MoeAuxLossConfig = Field(default_factory=MoeAuxLossConfig)


class ScheduleConfig(StrictBaseModel):
    type: Literal["linear", "cosine", "constant"] = "linear"
    start: float = 1.0
    end: float = 0.75


class MTPConfig(StrictBaseModel):
    enabled: bool = True
    depths: int | Literal["adapter_default"] = "adapter_default"
    beta_schedule: ScheduleConfig = Field(default_factory=lambda: ScheduleConfig(type="cosine", start=0.3, end=0.1))


class KDConfig(StrictBaseModel):
    enabled: bool = True
    teacher_mode: Literal[
        "online_full_logits",
        "offline_full_logits_cache",
        "offline_topk_logit_cache",
        "remote_worker_full_logits",
        "openai_api",
    ] = "online_full_logits"
    offline_full_logits_cache_path: str | None = None
    temperature: float = 1.0
    lambda_schedule: ScheduleConfig = Field(default_factory=lambda: ScheduleConfig(type="linear", start=1.0, end=0.75))
    mtp: MTPConfig = Field(default_factory=MTPConfig)


class QuantProtectConfig(StrictBaseModel):
    router: str = "bf16"
    norms: str = "bf16"
    embeddings: str = "bf16"
    shared_experts: str = "int8_or_bf16"


class QuantizationConfig(StrictBaseModel):
    enabled: bool = False
    mode: Literal["augmented_saliency_mixed_precision", "none"] = "augmented_saliency_mixed_precision"
    apply_stage: Literal["post_distill", "during_training"] = "post_distill"
    target_avg_bits: float | None = None
    prune_shared_experts: bool = False
    protect: QuantProtectConfig = Field(default_factory=QuantProtectConfig)


class PerplexityConfig(StrictBaseModel):
    enabled: bool = True
    dataset_name: str = "synthetic"
    split: str = "test"


class EvaluationConfig(StrictBaseModel):
    eval_every_steps: int = 10000
    perplexity: PerplexityConfig = Field(default_factory=PerplexityConfig)
    tasks: list[str] = Field(default_factory=list)
    smoke_prompts: bool = True
    speculative_acceptance: bool = True


class RuntimeLocalConfig(StrictBaseModel):
    num_gpus: int | Literal["auto"] = "auto"
    allow_full_model_run: bool = False


class RuntimeSSHConfig(StrictBaseModel):
    host: str | None = None
    user: str | None = None
    port: int = 22
    key_path: str | None = None
    dry_run: bool = True


class RuntimeSkyPilotConfig(StrictBaseModel):
    cluster_name: str = "slimder"
    accelerators: str = "H100:8"
    cloud: str = "auto"
    region: str | None = None
    image_id: str | None = None
    disk_size_gb: int = 512
    autostop_minutes: int = 60
    dry_run: bool = True


class RuntimeWorkerConfig(StrictBaseModel):
    api_url: str | None = None
    auth_token: str | None = None
    auth_token_env: str | None = "SLIMDER_WORKER_TOKEN"
    timeout_seconds: float = 60.0


class TrackingConfig(StrictBaseModel):
    backend: Literal["tensorboard", "wandb", "mlflow", "none"] = "tensorboard"


class RuntimeConfig(StrictBaseModel):
    backend: Literal["local", "ssh", "skypilot", "worker"] = "local"
    local: RuntimeLocalConfig = Field(default_factory=RuntimeLocalConfig)
    ssh: RuntimeSSHConfig = Field(default_factory=RuntimeSSHConfig)
    skypilot: RuntimeSkyPilotConfig = Field(default_factory=RuntimeSkyPilotConfig)
    worker: RuntimeWorkerConfig = Field(default_factory=RuntimeWorkerConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


class SlimderConfig(StrictBaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    progressive: ProgressiveConfig = Field(default_factory=ProgressiveConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    kd: KDConfig = Field(default_factory=KDConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("project")
    @classmethod
    def ensure_project(cls, v: ProjectConfig, info: ValidationInfo) -> ProjectConfig:
        return v

    @model_validator(mode="after")
    def validate_implemented_target_fields(self) -> "SlimderConfig":
        target = self.compression.target
        errors: list[str] = []
        if target.total_param_reduction is not None:
            errors.append("compression.target.total_param_reduction is planner-only and is not accepted by direct compression yet; use a preset or explicit target dimensions")
        if target.active_param_reduction is not None:
            errors.append("compression.target.active_param_reduction is planner-only and is not accepted by direct compression yet; use a preset or explicit target dimensions")
        if target.depth_remove_fraction is not None and not 0 <= target.depth_remove_fraction < 1:
            errors.append("compression.target.depth_remove_fraction must be >= 0 and < 1")
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @model_validator(mode="after")
    def validate_paper_faithful(self) -> "SlimderConfig":
        if not self.project.paper_faithful:
            return self
        errors: list[str] = []
        if self.compression.depth.method != "last_layers":
            errors.append("paper_faithful requires depth.method=last_layers")
        if self.compression.width.method != "rmsnorm_mean_abs":
            errors.append("paper_faithful requires width.method=rmsnorm_mean_abs")
        if self.compression.experts.base_selection != "next_highest_importance":
            errors.append("paper_faithful requires experts.base_selection=next_highest_importance")
        if self.compression.experts.router_row_strategy != "base":
            errors.append("paper_faithful requires experts.router_row_strategy=base")
        if self.compression.experts.method != "partial_preservation_merge":
            errors.append("paper_faithful requires experts.method=partial_preservation_merge")
        if self.kd.teacher_mode == "offline_topk_logit_cache":
            errors.append("paper_faithful rejects offline_topk_logit_cache")
        if self.kd.teacher_mode == "openai_api":
            errors.append("paper_faithful rejects generic OpenAI-compatible teacher APIs")
        if not self.compression.experts.keep_shared_experts or self.compression.target.shared_experts != "keep":
            errors.append("paper_faithful requires keeping shared experts")
        if self.quantization.enabled:
            errors.append("paper_faithful rejects saliency mixed-precision quantization")
        if self.quantization.prune_shared_experts:
            errors.append("paper_faithful rejects shared expert pruning")
        if self.kd.lambda_schedule.type != "linear" or self.kd.lambda_schedule.start != 1.0 or self.kd.lambda_schedule.end != 0.75:
            errors.append("paper_faithful requires lambda linear 1.0->0.75")
        beta = self.kd.mtp.beta_schedule
        if beta.type != "cosine" or beta.start != 0.3 or beta.end != 0.1:
            errors.append("paper_faithful requires beta cosine 0.3->0.1")
        if self.quantization.apply_stage == "during_training":
            errors.append("paper_faithful rejects quantization during training")
        if errors:
            raise ValueError("; ".join(errors))
        return self


def load_config(path: str | Path) -> SlimderConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SlimderConfig.model_validate(data)


def save_config(config: SlimderConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json")
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

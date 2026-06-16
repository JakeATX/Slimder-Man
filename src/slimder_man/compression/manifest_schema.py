from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ManifestTarget(StrictModel):
    hidden_size: int
    remove_last_n_layers: int
    routed_experts: int
    top_k: int


class ManifestDepth(StrictModel):
    method: str
    kept_block_indices: list[int]


class ManifestWidth(StrictModel):
    method: str
    hidden_keep_indices: list[int]
    hidden_size_before: int
    hidden_size_after: int


class ManifestExpertLayer(StrictModel):
    layer_idx: int
    s_keep: list[int]
    s_base: list[int]
    groups: dict[str, list[int]]
    new_expert_order: list[int]
    warning: str | None = None

    @model_validator(mode="after")
    def all_sets_valid(self) -> "ManifestExpertLayer":
        if len(self.new_expert_order) != len(self.s_keep) + len(self.s_base):
            raise ValueError("new_expert_order must contain keep plus base experts")
        return self


class ManifestExperts(StrictModel):
    method: str
    importance_metric: str
    similarity_metric: str
    layers: list[ManifestExpertLayer]


class ManifestRouter(StrictModel):
    row_strategy: str
    top_k_before: int
    top_k_after: int


class ManifestParamCounts(StrictModel):
    before: int
    after: int
    actual_after: int | None = None


class CompressionManifest(StrictModel):
    schema_version: str = "1.0"
    paper_faithful: bool
    teacher_model: str
    teacher_revision: str | None = None
    seed: int
    calibration: dict
    target: ManifestTarget
    depth: ManifestDepth
    width: ManifestWidth
    experts: ManifestExperts
    router: ManifestRouter
    param_counts: ManifestParamCounts
    artifact_hashes: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_shapes(self) -> "CompressionManifest":
        if self.width.hidden_size_after != self.target.hidden_size:
            raise ValueError("width.hidden_size_after must match target.hidden_size")
        if len(self.width.hidden_keep_indices) != self.target.hidden_size:
            raise ValueError("hidden_keep_indices length must match target.hidden_size")
        if self.router.top_k_after > self.target.routed_experts:
            raise ValueError("router.top_k_after exceeds target routed experts")
        return self

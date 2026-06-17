from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .base import ArchitectureInfo, MoELayerInfo, count_parameters, dtype_summary


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class TinyExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TinyMoE(nn.Module):
    def __init__(self, hidden_size: int, num_routed_experts: int, top_k: int, num_shared_experts: int, expert_intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_routed_experts, bias=False)
        self.experts = nn.ModuleList([TinyExpert(hidden_size, expert_intermediate_size) for _ in range(num_routed_experts)])
        self.shared_experts = nn.ModuleList([TinyExpert(hidden_size, expert_intermediate_size) for _ in range(num_shared_experts)])
        self.shared_gate = nn.Linear(hidden_size, num_shared_experts, bias=False) if num_shared_experts else None
        self.last_router_logits: torch.Tensor | None = None
        self.last_topk_indices: torch.Tensor | None = None
        self.last_topk_weights: torch.Tensor | None = None
        self.last_expert_output_norm2: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.router(flat)
        k = min(self.top_k, self.num_routed_experts)
        topv, topi = torch.topk(logits, k=k, dim=-1)
        weights = torch.softmax(topv, dim=-1)
        out = torch.zeros_like(flat)
        norm2 = torch.zeros(flat.shape[0], self.num_routed_experts, device=flat.device, dtype=flat.dtype)
        for slot in range(k):
            indices = topi[:, slot]
            slot_w = weights[:, slot].unsqueeze(-1)
            for expert_idx, expert in enumerate(self.experts):
                mask = indices == expert_idx
                if mask.any():
                    y = expert(flat[mask])
                    out[mask] += slot_w[mask] * y
                    norm2[mask, expert_idx] = y.pow(2).sum(dim=-1)
        if self.shared_experts:
            if self.shared_gate is None:
                gate = torch.ones(flat.shape[0], len(self.shared_experts), device=flat.device, dtype=flat.dtype)
            else:
                gate = torch.sigmoid(self.shared_gate(flat))
            for i, expert in enumerate(self.shared_experts):
                out += gate[:, i : i + 1] * expert(flat)
        self.last_router_logits = logits.detach()
        self.last_topk_indices = topi.detach()
        self.last_topk_weights = weights.detach()
        self.last_expert_output_norm2 = norm2.detach()
        return out.reshape(shape)


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int, kind: str, num_routed_experts: int, top_k: int, num_shared_experts: int, expert_intermediate_size: int):
        super().__init__()
        self.kind = kind
        self.input_layernorm = TinyRMSNorm(hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.post_attention_layernorm = TinyRMSNorm(hidden_size)
        self.moe = TinyMoE(hidden_size, num_routed_experts, top_k, num_shared_experts, expert_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.input_layernorm(x))
        x = x + self.moe(self.post_attention_layernorm(x))
        return x


@dataclass
class TinyConfig:
    vocab_size: int = 128
    hidden_size: int = 16
    num_layers: int = 4
    block_pattern: tuple[str, ...] = ("linear_attention", "full_attention", "linear_attention", "full_attention")
    num_routed_experts: int = 8
    top_k: int = 2
    num_shared_experts: int = 1
    expert_intermediate_size: int = 32
    mtp_depths: int = 2
    tie_embeddings: bool = True


class TinyOutput:
    def __init__(self, logits: torch.Tensor, loss: torch.Tensor | None, mtp_logits: list[torch.Tensor], aux_loss: torch.Tensor | None = None):
        self.logits = logits
        self.loss = loss
        self.mtp_logits = mtp_logits
        self.aux_loss = aux_loss


class TinyMoEForCausalLM(nn.Module):
    def __init__(self, config: TinyConfig | None = None):
        super().__init__()
        rng_state = torch.get_rng_state()
        self.config = config or TinyConfig()
        cfg = self.config
        try:
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.layers = nn.ModuleList([
                TinyBlock(
                    cfg.hidden_size,
                    cfg.block_pattern[i % len(cfg.block_pattern)],
                    cfg.num_routed_experts,
                    cfg.top_k,
                    cfg.num_shared_experts,
                    cfg.expert_intermediate_size,
                )
                for i in range(cfg.num_layers)
            ])
            self.norm = TinyRMSNorm(cfg.hidden_size)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.embed_tokens.weight
            self.mtp_heads = nn.ModuleList([nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False) for _ in range(cfg.mtp_depths)])
            self._init_deterministic()
        finally:
            torch.set_rng_state(rng_state)

    def _init_deterministic(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(123)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02, generator=generator)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02, generator=generator)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> TinyOutput:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        mtp_logits = [head(x) for head in self.mtp_heads]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return TinyOutput(logits=logits, loss=loss, mtp_logits=mtp_logits)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 8) -> torch.Tensor:
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self(out).logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, next_id], dim=1)
        return out

    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        data = asdict(self.config)
        data["block_pattern"] = list(data["block_pattern"])
        (path / "config.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, output_dir: str | Path) -> "TinyMoEForCausalLM":
        path = Path(output_dir)
        data = json.loads((path / "config.json").read_text(encoding="utf-8"))
        data["block_pattern"] = tuple(data["block_pattern"])
        model = cls(TinyConfig(**data))
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu", weights_only=True))
        return model


class TinyAdapter:
    def match(self, model: nn.Module, config: object | None = None) -> bool:
        return isinstance(model, TinyMoEForCausalLM)

    def describe_architecture(self, model: TinyMoEForCausalLM, config: object | None = None) -> ArchitectureInfo:
        cfg = model.config
        blocks = list(model.layers)
        moe_infos = [
            MoELayerInfo(i, layer.moe.num_routed_experts, len(layer.moe.shared_experts), layer.moe.top_k)
            for i, layer in enumerate(blocks)
        ]
        return ArchitectureInfo(
            model_type="tiny_moe",
            total_params=count_parameters(model),
            active_params_estimate=None,
            hidden_size=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
            num_layers=len(blocks),
            block_kinds=[b.kind for b in blocks],
            num_full_attention_layers=sum(1 for b in blocks if b.kind == "full_attention"),
            num_linear_attention_layers=sum(1 for b in blocks if b.kind == "linear_attention"),
            moe_layers=moe_infos,
            has_mtp=len(model.mtp_heads) > 0,
            mtp_depths=len(model.mtp_heads),
            tied_embeddings=model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr(),
            dtype_summary=dtype_summary(model),
            tensor_name_map={name: name for name, _ in model.named_parameters()},
        )

    def iter_transformer_blocks(self, model: TinyMoEForCausalLM) -> list[nn.Module]:
        return list(model.layers)

    def get_block_kind(self, block: TinyBlock) -> str:
        return block.kind

    def iter_rmsnorms(self, model: TinyMoEForCausalLM) -> list[nn.Module]:
        norms: list[nn.Module] = []
        for block in model.layers:
            norms.extend([block.input_layernorm, block.post_attention_layernorm])
        norms.append(model.norm)
        return norms

    def iter_moe_layers(self, model: TinyMoEForCausalLM) -> list[TinyMoE]:
        return [block.moe for block in model.layers]

    def get_routed_experts(self, moe: TinyMoE) -> list[nn.Module]:
        return list(moe.experts)

    def get_shared_experts(self, moe: TinyMoE) -> list[nn.Module]:
        return list(moe.shared_experts)

    def get_router(self, moe: TinyMoE) -> nn.Linear:
        return moe.router

    def get_mtp_modules(self, model: TinyMoEForCausalLM) -> list[nn.Module]:
        return list(model.mtp_heads)

    def slice_hidden_channels(self, model: TinyMoEForCausalLM, keep_idx: torch.Tensor) -> None:
        from slimder_man.compression.width import slice_tiny_hidden

        slice_tiny_hidden(model, keep_idx)

    def drop_blocks(self, model: TinyMoEForCausalLM, keep_block_idx: list[int]) -> None:
        model.layers = nn.ModuleList([model.layers[i] for i in keep_block_idx])
        model.config.num_layers = len(model.layers)
        model.config.block_pattern = tuple(layer.kind for layer in model.layers)

    def replace_experts(self, moe: TinyMoE, new_experts: list[nn.Module], router_rows: torch.Tensor, new_top_k: int) -> None:
        moe.experts = nn.ModuleList(new_experts)
        moe.num_routed_experts = len(new_experts)
        moe.top_k = min(new_top_k, len(new_experts))
        new_router = nn.Linear(moe.hidden_size, len(new_experts), bias=False)
        with torch.no_grad():
            new_router.weight.copy_(router_rows)
        moe.router = new_router

    def update_config_after_compression(self, model: TinyMoEForCausalLM, manifest: dict) -> None:
        model.config.hidden_size = manifest["target"]["hidden_size"]
        model.config.num_layers = len(model.layers)
        model.config.num_routed_experts = manifest["target"]["routed_experts"]
        model.config.top_k = manifest["target"]["top_k"]

    def save_pretrained(self, model: TinyMoEForCausalLM, output_dir: str, manifest: dict | None = None) -> None:
        model.save_pretrained(output_dir)
        if manifest is not None:
            Path(output_dir, "compression_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def clone_tiny_model(model: TinyMoEForCausalLM) -> TinyMoEForCausalLM:
    return deepcopy(model)

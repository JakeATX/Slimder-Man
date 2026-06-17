from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DummyHfMoeConfig:
    model_type: str = "dummy_hf_moe"
    vocab_size: int = 257
    hidden_size: int = 32
    intermediate_size: int = 64
    num_hidden_layers: int = 3
    num_experts: int = 6
    num_experts_per_tok: int = 2
    num_shared_experts: int = 2
    tie_word_embeddings: bool = False


class DummyExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)))


class DummyHfMoeLayer(nn.Module):
    def __init__(self, config: DummyHfMoeConfig):
        super().__init__()
        self.num_experts_per_tok = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([DummyExpert(config.hidden_size, config.intermediate_size) for _ in range(config.num_experts)])
        self.shared_experts = nn.ModuleList([
            DummyExpert(config.hidden_size, config.intermediate_size) for _ in range(config.num_shared_experts)
        ])
        self.last_router_logits = None
        self.last_topk_indices = None
        self.last_topk_weights = None
        self.last_expert_output_norm2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.gate(flat)
        topv, topi = torch.topk(logits, k=self.num_experts_per_tok, dim=-1)
        topw = torch.softmax(topv, dim=-1)
        out = torch.zeros_like(flat)
        norm2 = torch.zeros(flat.shape[0], len(self.experts), device=flat.device, dtype=flat.dtype)
        for slot in range(self.num_experts_per_tok):
            for expert_idx, expert in enumerate(self.experts):
                mask = topi[:, slot] == expert_idx
                if mask.any():
                    y = expert(flat[mask])
                    out[mask] += topw[mask, slot : slot + 1] * y
                    norm2[mask, expert_idx] = y.pow(2).sum(-1)
        for expert in self.shared_experts:
            out += expert(flat)
        self.last_router_logits = logits.detach()
        self.last_topk_indices = topi.detach()
        self.last_topk_weights = topw.detach()
        self.last_expert_output_norm2 = norm2.detach()
        return out.reshape(shape)


class DummyAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class DummyHfBlock(nn.Module):
    def __init__(self, config: DummyHfMoeConfig):
        super().__init__()
        self.self_attn = DummyAttention(config.hidden_size)
        self.mlp = DummyHfMoeLayer(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class DummyBackbone(nn.Module):
    def __init__(self, config: DummyHfMoeConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DummyHfBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DummyHfMoeForCausalLM(nn.Module):
    def __init__(self, config: DummyHfMoeConfig | None = None):
        super().__init__()
        self.config = config or DummyHfMoeConfig()
        self.model = DummyBackbone(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self) -> None:
        torch.manual_seed(2026)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> object:
        hidden = self.model(input_ids)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1))
        return type("DummyCausalLMOutput", (), {"logits": logits, "loss": loss, "mtp_logits": []})()

    def save_pretrained(self, output_dir: str | Path, safe_serialization: bool = True) -> None:
        from safetensors.torch import save_file

        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        if safe_serialization:
            save_file(self.state_dict(), path / "model.safetensors")
        else:
            torch.save(self.state_dict(), path / "pytorch_model.bin")
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, output_dir: str | Path):
        from safetensors.torch import load_file

        path = Path(output_dir)
        model = cls(DummyHfMoeConfig(**json.loads((path / "config.json").read_text(encoding="utf-8"))))
        state = load_file(path / "model.safetensors") if (path / "model.safetensors").exists() else torch.load(path / "pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state)
        return model


class DummyTokenizer:
    vocab_size = 257

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        pieces = text.strip().split()
        return [sum(ord(ch) for ch in piece) % self.vocab_size for piece in pieces]

    def __call__(self, texts, max_length: int, padding: str = "max_length", truncation: bool = True, return_tensors: str = "pt"):
        rows = []
        for text in texts:
            ids = self.encode(str(text))
            if truncation:
                ids = ids[:max_length]
            if padding == "max_length":
                ids = ids + [0] * (max_length - len(ids))
            rows.append(ids)
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}

    def save_pretrained(self, output_dir: str | Path) -> None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "tokenizer_config.json").write_text(json.dumps({"type": "dummy"}), encoding="utf-8")


DummyMoeConfig = DummyHfMoeConfig
HfDummyMoeForCausalLM = DummyHfMoeForCausalLM

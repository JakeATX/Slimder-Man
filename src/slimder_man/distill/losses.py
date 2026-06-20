from __future__ import annotations

import torch
from torch.nn import functional as F

from .mtp import align_mtp_training_tensors


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    aligned = mask.to(device=values.device, dtype=values.dtype)
    while aligned.ndim < values.ndim:
        aligned = aligned.unsqueeze(-1)
    denom = aligned.sum().clamp_min(1.0)
    return (values * aligned).sum() / denom


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    s = student_logits[:, :-1, :] / temperature
    t = teacher_logits[:, :-1, :] / temperature
    q = torch.softmax(t, dim=-1)
    log_p = torch.log_softmax(s, dim=-1)
    mask = attention_mask[:, 1:] if attention_mask is not None else None
    return _masked_mean(-(q * log_p).sum(dim=-1), mask) * (temperature**2)


def lm_loss(student_logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
    token_losses = F.cross_entropy(
        student_logits[:, :-1, :].reshape(-1, student_logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape(input_ids.shape[0], -1)
    mask = attention_mask[:, 1:] if attention_mask is not None else None
    return _masked_mean(token_losses, mask)


def mtp_losses(
    student_mtp_logits: list[torch.Tensor],
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    lm_terms = []
    kd_terms = []
    for i, logits in enumerate(student_mtp_logits, start=1):
        if logits.shape[1] <= i:
            continue
        aligned, labels, teacher_aligned = align_mtp_training_tensors(logits, input_ids, teacher_logits, i)
        label_mask = attention_mask[:, i:] if attention_mask is not None else None
        token_losses = F.cross_entropy(aligned.reshape(-1, aligned.shape[-1]), labels.reshape(-1), reduction="none").reshape(labels.shape)
        lm_terms.append(_masked_mean(token_losses, label_mask))
        q = torch.softmax(teacher_aligned / temperature, dim=-1)
        log_p = torch.log_softmax(aligned / temperature, dim=-1)
        kd_terms.append(_masked_mean(-(q * log_p).sum(dim=-1), label_mask) * (temperature**2))
    zero = teacher_logits.sum() * 0
    return (torch.stack(lm_terms).mean() if lm_terms else zero, torch.stack(kd_terms).mean() if kd_terms else zero)


def total_distill_loss(
    student_out,
    teacher_out,
    input_ids: torch.Tensor,
    lambda_t: float,
    beta_t: float,
    temperature: float = 1.0,
    kd_enabled: bool = True,
    mtp_enabled: bool = True,
    moe_aux_weight: float = 0.0,
    attention_mask: torch.Tensor | None = None,
):
    l_lm = lm_loss(student_out.logits, input_ids, attention_mask=attention_mask)
    zero = l_lm * 0
    l_kd = kd_loss(student_out.logits, teacher_out.logits, temperature, attention_mask=attention_mask) if kd_enabled else zero
    l_mtp_lm, l_mtp_kd = mtp_losses(student_out.mtp_logits, teacher_out.logits, input_ids, temperature, attention_mask=attention_mask) if mtp_enabled else (zero, zero)
    l_moe_aux = _moe_aux_loss(student_out, zero) if moe_aux_weight != 0.0 else zero
    effective_lambda = lambda_t if kd_enabled else 0.0
    effective_beta = beta_t if mtp_enabled else 0.0
    total = (
        (1 - effective_lambda) * l_lm
        + effective_lambda * l_kd
        + effective_beta * ((1 - effective_lambda) * l_mtp_lm + effective_lambda * l_mtp_kd)
        + moe_aux_weight * l_moe_aux
    )
    return total, {
        "loss_lm": float(l_lm.detach()),
        "loss_kd": float(l_kd.detach()),
        "loss_mtp_lm": float(l_mtp_lm.detach()),
        "loss_mtp_kd": float(l_mtp_kd.detach()),
        "loss_moe_aux": float(l_moe_aux.detach()),
        "moe_aux_weight": float(moe_aux_weight),
    }


def _moe_aux_loss(student_out, zero: torch.Tensor) -> torch.Tensor:
    for name in ("aux_loss", "router_aux_loss", "moe_aux_loss"):
        value = getattr(student_out, name, None)
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            value = value.to(device=zero.device, dtype=zero.dtype)
            return value.mean() if value.ndim > 0 else value
        return zero + float(value)
    return zero

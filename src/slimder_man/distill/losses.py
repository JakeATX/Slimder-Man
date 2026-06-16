from __future__ import annotations

import torch
from torch.nn import functional as F

from .mtp import align_mtp_logits, mtp_labels


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    s = student_logits[:, :-1, :] / temperature
    t = teacher_logits[:, :-1, :] / temperature
    q = torch.softmax(t, dim=-1)
    log_p = torch.log_softmax(s, dim=-1)
    return -(q * log_p).sum(dim=-1).mean()


def lm_loss(student_logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(student_logits[:, :-1, :].reshape(-1, student_logits.shape[-1]), input_ids[:, 1:].reshape(-1))


def mtp_losses(student_mtp_logits: list[torch.Tensor], teacher_logits: torch.Tensor, input_ids: torch.Tensor, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    lm_terms = []
    kd_terms = []
    for i, logits in enumerate(student_mtp_logits, start=1):
        if logits.shape[1] <= i:
            continue
        aligned = align_mtp_logits(logits, i)
        labels = mtp_labels(input_ids, i)
        lm_terms.append(F.cross_entropy(aligned.reshape(-1, aligned.shape[-1]), labels.reshape(-1)))
        teacher_aligned = teacher_logits[:, i:, :]
        q = torch.softmax(teacher_aligned / temperature, dim=-1)
        log_p = torch.log_softmax(aligned / temperature, dim=-1)
        kd_terms.append(-(q * log_p).sum(dim=-1).mean())
    zero = teacher_logits.sum() * 0
    return (torch.stack(lm_terms).mean() if lm_terms else zero, torch.stack(kd_terms).mean() if kd_terms else zero)


def total_distill_loss(student_out, teacher_out, input_ids: torch.Tensor, lambda_t: float, beta_t: float, temperature: float = 1.0):
    l_lm = lm_loss(student_out.logits, input_ids)
    l_kd = kd_loss(student_out.logits, teacher_out.logits, temperature)
    l_mtp_lm, l_mtp_kd = mtp_losses(student_out.mtp_logits, teacher_out.logits, input_ids, temperature)
    total = (1 - lambda_t) * l_lm + lambda_t * l_kd + beta_t * ((1 - lambda_t) * l_mtp_lm + lambda_t * l_mtp_kd)
    return total, {"loss_lm": float(l_lm.detach()), "loss_kd": float(l_kd.detach()), "loss_mtp_lm": float(l_mtp_lm.detach()), "loss_mtp_kd": float(l_mtp_kd.detach())}

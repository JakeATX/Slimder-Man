from __future__ import annotations

import torch


def mtp_labels(input_ids: torch.Tensor, depth: int) -> torch.Tensor:
    if depth <= 0:
        raise ValueError("depth must be positive")
    return input_ids[:, depth:]


def align_mtp_logits(logits: torch.Tensor, depth: int) -> torch.Tensor:
    if depth <= 0:
        raise ValueError("depth must be positive")
    return logits[:, : logits.shape[1] - depth, :]


def align_teacher_logits_for_mtp(teacher_logits: torch.Tensor, depth: int) -> torch.Tensor:
    if depth <= 0:
        raise ValueError("depth must be positive")
    return teacher_logits[:, depth - 1 : teacher_logits.shape[1] - 1, :]

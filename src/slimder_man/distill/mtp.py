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


def align_mtp_training_tensors(
    student_logits: torch.Tensor,
    input_ids: torch.Tensor,
    teacher_logits: torch.Tensor,
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align MTP tensors so every row targets token p + depth from source p."""
    student_aligned = align_mtp_logits(student_logits, depth)
    labels = mtp_labels(input_ids, depth)
    teacher_aligned = align_teacher_logits_for_mtp(teacher_logits, depth)
    if student_aligned.shape[1] != labels.shape[1] or teacher_aligned.shape[1] != labels.shape[1]:
        raise ValueError("aligned MTP student logits, labels, and teacher logits must have matching sequence lengths")
    return student_aligned, labels, teacher_aligned

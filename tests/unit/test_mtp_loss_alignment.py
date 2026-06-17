import torch

from slimder_man.distill.mtp import align_mtp_logits, align_teacher_logits_for_mtp, mtp_labels


def test_mtp_alignment():
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    assert mtp_labels(ids, 1).tolist() == [[11, 12, 13, 14]]
    assert mtp_labels(ids, 2).tolist() == [[12, 13, 14]]
    logits = torch.zeros(1, 5, 128)
    assert align_mtp_logits(logits, 1).shape[1] == 4
    assert align_mtp_logits(logits, 2).shape[1] == 3


def test_mtp_teacher_logits_align_to_next_token_distribution():
    teacher_positions = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    assert align_teacher_logits_for_mtp(teacher_positions, 1).squeeze(-1).tolist() == [[0, 1, 2, 3]]
    assert align_teacher_logits_for_mtp(teacher_positions, 2).squeeze(-1).tolist() == [[1, 2, 3]]

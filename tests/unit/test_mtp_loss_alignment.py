import torch

from slimder_man.distill.mtp import align_mtp_logits, align_mtp_training_tensors, align_teacher_logits_for_mtp, mtp_labels


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


def test_mtp_training_tensors_align_teacher_distribution_to_same_future_tokens():
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    vocab = 16
    teacher = torch.full((1, 5, vocab), -20.0)
    for pos, token in enumerate(ids[0, 1:].tolist()):
        teacher[0, pos, token] = 20.0
    student_mtp = torch.zeros(1, 5, vocab)

    _, depth_1_labels, depth_1_teacher = align_mtp_training_tensors(student_mtp, ids, teacher, 1)
    _, depth_2_labels, depth_2_teacher = align_mtp_training_tensors(student_mtp, ids, teacher, 2)

    assert depth_1_labels.tolist() == [[11, 12, 13, 14]]
    assert depth_1_teacher.argmax(dim=-1).tolist() == depth_1_labels.tolist()
    assert depth_2_labels.tolist() == [[12, 13, 14]]
    assert depth_2_teacher.argmax(dim=-1).tolist() == depth_2_labels.tolist()


def test_mtp_training_tensors_reject_misaligned_sequence_lengths():
    ids = torch.tensor([[10, 11, 12, 13]])
    student_mtp = torch.zeros(1, 3, 16)
    teacher = torch.zeros(1, 4, 16)

    try:
        align_mtp_training_tensors(student_mtp, ids, teacher, 1)
    except ValueError as exc:
        assert "matching sequence lengths" in str(exc)
    else:
        raise AssertionError("misaligned MTP tensors should fail explicitly")

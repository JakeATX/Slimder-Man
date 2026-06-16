import torch

from slimder_man.distill.mtp import align_mtp_logits, mtp_labels


def test_mtp_alignment():
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    assert mtp_labels(ids, 1).tolist() == [[11, 12, 13, 14]]
    assert mtp_labels(ids, 2).tolist() == [[12, 13, 14]]
    logits = torch.zeros(1, 5, 128)
    assert align_mtp_logits(logits, 1).shape[1] == 4
    assert align_mtp_logits(logits, 2).shape[1] == 3

from types import SimpleNamespace

import torch

from slimder_man.distill.losses import lm_loss, total_distill_loss


def _out(logits: torch.Tensor, mtp_logits: list[torch.Tensor] | None = None):
    return SimpleNamespace(logits=logits, mtp_logits=mtp_logits or [])


def test_kd_disabled_uses_lm_loss_even_when_lambda_is_one():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    student_logits = torch.zeros(1, 4, 5)
    teacher_logits = torch.zeros(1, 4, 5)
    teacher_logits[..., 4] = 20

    total, parts = total_distill_loss(
        _out(student_logits),
        _out(teacher_logits),
        input_ids,
        lambda_t=1.0,
        beta_t=0.0,
        kd_enabled=False,
    )

    assert torch.allclose(total, lm_loss(student_logits, input_ids))
    assert parts["loss_kd"] == 0.0


def test_mtp_disabled_ignores_beta_and_mtp_logits():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    student_logits = torch.zeros(1, 4, 5)
    teacher_logits = torch.zeros(1, 4, 5)
    bad_mtp = [torch.full((1, 4, 5), -100.0)]

    total, parts = total_distill_loss(
        _out(student_logits, bad_mtp),
        _out(teacher_logits),
        input_ids,
        lambda_t=0.0,
        beta_t=100.0,
        mtp_enabled=False,
    )

    assert torch.allclose(total, lm_loss(student_logits, input_ids))
    assert parts["loss_mtp_lm"] == 0.0
    assert parts["loss_mtp_kd"] == 0.0

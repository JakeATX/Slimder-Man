from types import SimpleNamespace

import pytest
import torch

from slimder_man.distill.losses import kd_loss, lm_loss, mtp_losses, total_distill_loss


def _out(logits: torch.Tensor, mtp_logits: list[torch.Tensor] | None = None, aux_loss=None):
    return SimpleNamespace(logits=logits, mtp_logits=mtp_logits or [], aux_loss=aux_loss)


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


def test_kd_loss_applies_temperature_squared_scaling():
    student = torch.tensor([[[0.2, -0.1, 0.4], [0.0, 0.5, -0.2]]])
    teacher = torch.tensor([[[1.2, -0.4, 0.1], [0.3, 0.7, -0.8]]])
    temperature = 2.5

    softened = -(torch.softmax(teacher[:, :-1, :] / temperature, dim=-1) * torch.log_softmax(student[:, :-1, :] / temperature, dim=-1)).sum(dim=-1).mean()

    assert torch.allclose(kd_loss(student, teacher, temperature), softened * temperature**2)


def test_lm_and_kd_losses_ignore_padding_with_attention_mask():
    input_ids = torch.tensor([[1, 2, 3, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    student = torch.zeros(1, 4, 5)
    teacher = torch.zeros(1, 4, 5)
    student[:, 2, 0] = -50.0
    teacher[:, 2, 4] = 50.0

    expected_lm = torch.nn.functional.cross_entropy(student[:, :2, :].reshape(-1, 5), input_ids[:, 1:3].reshape(-1))
    expected_kd = kd_loss(student[:, :3, :], teacher[:, :3, :], temperature=1.0)

    assert torch.allclose(lm_loss(student, input_ids, attention_mask=attention_mask), expected_lm)
    assert torch.allclose(kd_loss(student, teacher, attention_mask=attention_mask), expected_kd)


def test_mtp_kd_loss_applies_temperature_squared_scaling():
    ids = torch.tensor([[0, 1, 2, 1]])
    student_mtp = [torch.tensor([[[0.1, 0.2, -0.1], [0.0, 0.3, -0.2], [0.4, -0.2, 0.1], [0.2, -0.3, 0.5]]])]
    teacher = torch.tensor([[[1.0, 0.0, -0.5], [0.4, 0.2, -0.1], [0.1, 0.9, -0.4], [0.3, -0.2, 0.8]]])
    temperature = 3.0

    _, mtp_kd = mtp_losses(student_mtp, teacher, ids, temperature)
    teacher_aligned = teacher[:, :3, :]
    student_aligned = student_mtp[0][:, :-1, :]
    softened = -(torch.softmax(teacher_aligned / temperature, dim=-1) * torch.log_softmax(student_aligned / temperature, dim=-1)).sum(dim=-1).mean()

    assert torch.allclose(mtp_kd, softened * temperature**2)


def test_kd_temperature_must_be_positive():
    logits = torch.zeros(1, 3, 4)

    with pytest.raises(ValueError, match="temperature must be positive"):
        kd_loss(logits, logits, 0.0)
    with pytest.raises(ValueError, match="temperature must be positive"):
        mtp_losses([logits], logits, torch.tensor([[0, 1, 2]]), -1.0)


def test_moe_aux_loss_is_weighted_and_logged():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    student_logits = torch.zeros(1, 4, 5)
    teacher_logits = torch.zeros(1, 4, 5)

    base, base_parts = total_distill_loss(
        _out(student_logits),
        _out(teacher_logits),
        input_ids,
        lambda_t=0.0,
        beta_t=0.0,
        kd_enabled=False,
        mtp_enabled=False,
    )
    total, parts = total_distill_loss(
        _out(student_logits, aux_loss=torch.tensor(2.0)),
        _out(teacher_logits),
        input_ids,
        lambda_t=0.0,
        beta_t=0.0,
        kd_enabled=False,
        mtp_enabled=False,
        moe_aux_weight=0.25,
    )

    assert torch.allclose(total, base + 0.5)
    assert base_parts["loss_moe_aux"] == 0.0
    assert parts["loss_moe_aux"] == 2.0
    assert parts["moe_aux_weight"] == 0.25


def test_zero_weight_moe_aux_is_inert_even_when_nonfinite_or_vector():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    student_logits = torch.zeros(1, 4, 5)
    teacher_logits = torch.zeros(1, 4, 5)

    total, parts = total_distill_loss(
        _out(student_logits, aux_loss=torch.tensor([float("nan"), 1.0])),
        _out(teacher_logits),
        input_ids,
        lambda_t=0.0,
        beta_t=0.0,
        kd_enabled=False,
        mtp_enabled=False,
        moe_aux_weight=0.0,
    )

    assert torch.isfinite(total)
    assert parts["loss_moe_aux"] == 0.0


def test_vector_moe_aux_loss_is_reduced_when_enabled():
    input_ids = torch.tensor([[0, 1, 2, 3]])
    student_logits = torch.zeros(1, 4, 5)
    teacher_logits = torch.zeros(1, 4, 5)

    total, parts = total_distill_loss(
        _out(student_logits, aux_loss=torch.tensor([1.0, 3.0])),
        _out(teacher_logits),
        input_ids,
        lambda_t=0.0,
        beta_t=0.0,
        kd_enabled=False,
        mtp_enabled=False,
        moe_aux_weight=0.5,
    )

    assert torch.isfinite(total)
    assert parts["loss_moe_aux"] == 2.0
    assert parts["moe_aux_weight"] == 0.5

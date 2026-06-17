import torch

from slimder_man.adapters.hf_dummy import DummyHfMoeForCausalLM
from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.train_loop import train_tiny_distill


def test_tiny_model_initialization_does_not_mutate_global_torch_rng():
    torch.manual_seed(999)
    expected_first = torch.rand(4)
    expected_second = torch.rand(4)

    torch.manual_seed(999)
    actual_first = torch.rand(4)
    _ = TinyMoEForCausalLM()
    actual_second = torch.rand(4)

    assert torch.equal(actual_first, expected_first)
    assert torch.equal(actual_second, expected_second)


def test_hf_dummy_fixture_initialization_does_not_mutate_global_torch_rng():
    torch.manual_seed(999)
    expected_first = torch.rand(4)
    expected_second = torch.rand(4)

    torch.manual_seed(999)
    actual_first = torch.rand(4)
    _ = DummyHfMoeForCausalLM()
    actual_second = torch.rand(4)

    assert torch.equal(actual_first, expected_first)
    assert torch.equal(actual_second, expected_second)


def test_tiny_distill_is_reproducible_for_same_project_seed(tmp_path):
    cfg = SlimderConfig(
        project={"seed": 2026},
        calibration={"sample_count": 4, "sequence_length": 8},
        training={"train_steps": 2, "warmup_steps": 0},
    )

    teacher_a = TinyMoEForCausalLM()
    student_a = TinyMoEForCausalLM()
    result_a = train_tiny_distill(teacher_a, student_a, cfg, tmp_path / "a")

    teacher_b = TinyMoEForCausalLM()
    student_b = TinyMoEForCausalLM()
    result_b = train_tiny_distill(teacher_b, student_b, cfg, tmp_path / "b")

    assert [round(row["loss"], 8) for row in result_a["logs"]] == [round(row["loss"], 8) for row in result_b["logs"]]
    for param_a, param_b in zip(student_a.parameters(), student_b.parameters(), strict=True):
        assert torch.equal(param_a, param_b)

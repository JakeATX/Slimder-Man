import json
import sys
from pathlib import Path

import pytest

from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.train_loop import train_causal_lm_distill

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyHfMoeForCausalLM


class RecordingDummyHfMoeForCausalLM(DummyHfMoeForCausalLM):
    def __init__(self):
        super().__init__()
        self.forward_batch_sizes: list[int] = []

    def forward(self, input_ids, labels=None):
        self.forward_batch_sizes.append(int(input_ids.shape[0]))
        return super().forward(input_ids, labels=labels)


class AuxLossDummyHfMoeForCausalLM(DummyHfMoeForCausalLM):
    def forward(self, input_ids, labels=None):
        out = super().forward(input_ids, labels=labels)
        return type(
            "AuxDummyCausalLMOutput",
            (),
            {
                "logits": out.logits,
                "loss": out.loss,
                "mtp_logits": out.mtp_logits,
                "aux_loss": out.logits.sum() * 0 + 3.0,
            },
        )()


def test_generic_distill_resume_restores_state_with_fresh_model_instance(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = DummyHfMoeForCausalLM()
    cfg_first = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        calibration={"sample_count": 4, "sequence_length": 8},
        training={"train_steps": 2, "warmup_steps": 0},
    )
    batches, _ = sample_calibration_tokens(cfg_first.calibration, vocab_size=student.config.vocab_size)
    out_dir = tmp_path / "training"

    first = train_causal_lm_distill(teacher, student, cfg_first, out_dir, batches)
    assert first["global_step"] == 2
    assert first["gradient_accumulation_steps"] == 2
    state_after_first = json.loads((out_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert state_after_first["global_step"] == 2
    assert state_after_first["dataloader_position"] == 0
    assert state_after_first["gradient_accumulation_steps"] == 2
    assert (out_dir / "optimizer.pt").exists()
    assert (out_dir / "rng_state.pt").exists()

    resumed_student = DummyHfMoeForCausalLM.from_pretrained(out_dir / "resume_model")
    cfg_resume = cfg_first.model_copy(update={"training": cfg_first.training.model_copy(update={"train_steps": 4})})
    resumed = train_causal_lm_distill(teacher, resumed_student, cfg_resume, out_dir, batches, resume=True)

    assert resumed["global_step"] == 4
    assert [row["step"] for row in resumed["logs"]] == [1, 2, 3, 4]
    state_after_resume = json.loads((out_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert state_after_resume["global_step"] == 4
    assert state_after_resume["dataloader_position"] == 0
    assert len(state_after_resume["logs"]) == 4


def test_generic_distill_resume_uses_stage_step_when_global_offset_is_nonzero(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = DummyHfMoeForCausalLM()
    cfg_first = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        calibration={"sample_count": 4, "sequence_length": 8},
        training={"train_steps": 2, "warmup_steps": 0},
    )
    batches, _ = sample_calibration_tokens(cfg_first.calibration, vocab_size=student.config.vocab_size)
    out_dir = tmp_path / "offset_training"

    first = train_causal_lm_distill(
        teacher,
        student,
        cfg_first,
        out_dir,
        batches,
        global_step_offset=5,
        global_total_steps=9,
    )
    assert first["global_step"] == 7
    assert [row["step"] for row in first["logs"]] == [6, 7]
    state_after_first = json.loads((out_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert state_after_first["global_step"] == 7
    assert state_after_first["stage_step"] == 2

    resumed_student = DummyHfMoeForCausalLM.from_pretrained(out_dir / "resume_model")
    cfg_resume = cfg_first.model_copy(update={"training": cfg_first.training.model_copy(update={"train_steps": 4})})
    resumed = train_causal_lm_distill(
        teacher,
        resumed_student,
        cfg_resume,
        out_dir,
        batches,
        resume=True,
        global_step_offset=5,
        global_total_steps=9,
    )

    assert resumed["global_step"] == 9
    assert [row["step"] for row in resumed["logs"]] == [6, 7, 8, 9]
    assert [row["stage_step"] for row in resumed["logs"]] == [1, 2, 3, 4]
    state_after_resume = json.loads((out_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert state_after_resume["global_step"] == 9
    assert state_after_resume["stage_step"] == 4


def test_generic_distill_derives_steps_from_token_budget_and_accumulates_microbatches(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = DummyHfMoeForCausalLM()
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        calibration={"sample_count": 4, "sequence_length": 8},
        training={
            "train_steps": 0,
            "token_budget": 32,
            "global_batch_size": 2,
            "micro_batch_size": 1,
            "sequence_length": 8,
            "warmup_steps": 0,
        },
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    result = train_causal_lm_distill(teacher, student, cfg, tmp_path / "budget_training", batches)
    state = json.loads((tmp_path / "budget_training" / "trainer_state.json").read_text(encoding="utf-8"))

    assert result["global_step"] == 2
    assert result["stage_steps"] == 2
    assert result["gradient_accumulation_steps"] == 2
    assert [row["gradient_accumulation_steps"] for row in result["logs"]] == [2, 2]
    assert state["dataloader_position"] == 0


def test_generic_distill_micro_batch_size_controls_forward_batch_shape(tmp_path: Path):
    teacher = RecordingDummyHfMoeForCausalLM()
    student = RecordingDummyHfMoeForCausalLM()
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        calibration={"sample_count": 4, "sequence_length": 8},
        training={
            "train_steps": 1,
            "global_batch_size": 4,
            "micro_batch_size": 2,
            "sequence_length": 8,
            "warmup_steps": 0,
        },
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    result = train_causal_lm_distill(teacher, student, cfg, tmp_path / "microbatch_training", batches)
    state = json.loads((tmp_path / "microbatch_training" / "trainer_state.json").read_text(encoding="utf-8"))

    assert result["gradient_accumulation_steps"] == 2
    assert result["micro_batch_size"] == 2
    assert teacher.forward_batch_sizes == [2, 2]
    assert student.forward_batch_sizes == [2, 2]
    assert state["dataloader_position"] == 0


def test_generic_distill_rejects_unsupported_teacher_modes(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = DummyHfMoeForCausalLM()
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        kd={"teacher_mode": "offline_full_logits_cache"},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    with pytest.raises(ValueError, match="online_full_logits only"):
        train_causal_lm_distill(teacher, student, cfg, tmp_path / "unsupported", batches)


def test_generic_distill_logs_moe_aux_loss_when_model_exposes_it(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = AuxLossDummyHfMoeForCausalLM()
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        calibration={"sample_count": 2, "sequence_length": 8},
        training={
            "train_steps": 1,
            "global_batch_size": 1,
            "micro_batch_size": 1,
            "warmup_steps": 0,
            "moe_aux_loss": {"weight": 0.5},
        },
        kd={"enabled": False, "mtp": {"enabled": False}},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    result = train_causal_lm_distill(teacher, student, cfg, tmp_path / "aux_training", batches)

    row = result["logs"][0]
    assert row["loss_moe_aux"] == 3.0
    assert row["moe_aux_weight"] == 0.5
    assert row["loss"] == pytest.approx(row["loss_lm"] + 1.5, rel=1e-5)

import json
import sys
from pathlib import Path

from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.train_loop import train_causal_lm_distill

sys.path.append(str(Path(__file__).resolve().parents[1]))
from fixtures.hf_dummy_moe import DummyHfMoeForCausalLM


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
    state_after_first = json.loads((out_dir / "trainer_state.json").read_text(encoding="utf-8"))
    assert state_after_first["global_step"] == 2
    assert state_after_first["dataloader_position"] == 2
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

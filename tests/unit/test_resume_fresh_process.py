import json
import sys
from pathlib import Path

import pytest
import torch

from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.offline_cache import (
    OfflineFullLogitsCache,
    OfflineTopKLogitsCache,
    full_logits_cache_key,
    write_full_logits_cache,
    write_topk_logits_cache,
)
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


class RaisingTeacher(DummyHfMoeForCausalLM):
    def forward(self, input_ids, labels=None):
        raise AssertionError("remote_worker_full_logits must not call local teacher forward")


class FakeWorkerLogitsClient:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.calls: list[tuple[int, int]] = []

    def teacher_output(self, input_ids):
        self.calls.append(tuple(input_ids.shape))
        batch, seq = input_ids.shape
        logits = input_ids.new_zeros((batch, seq, self.vocab_size)).float()
        logits[..., 0] = 1.0
        return type("RemoteTeacherOutput", (), {"logits": logits})()


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
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        project={"paper_faithful": False, "output_dir": str(tmp_path)},
        kd={"teacher_mode": "openai_api"},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    with pytest.raises(ValueError, match="remote_worker_full_logits"):
        train_causal_lm_distill(teacher, student, cfg, tmp_path / "unsupported", batches)


def test_generic_distill_rejects_arbitrary_transformers_without_smoke_trainer_opt_in(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = DummyHfMoeForCausalLM()
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "org/tiny-moe"},
        training={"train_steps": 1},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    with pytest.raises(ValueError, match="allow_smoke_trainer=true"):
        train_causal_lm_distill(teacher, student, cfg, tmp_path / "smoke_guard", batches)


def test_generic_distill_rejects_missing_mtp_logits_in_paper_faithful_arbitrary_run(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = DummyHfMoeForCausalLM()
    cfg = SlimderConfig(
        project={"paper_faithful": True, "output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "org/no-mtp-moe"},
        training={"train_steps": 1, "global_batch_size": 1, "micro_batch_size": 1, "warmup_steps": 0, "allow_smoke_trainer": True},
        kd={"mtp": {"enabled": True}},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    with pytest.raises(ValueError, match="requires student MTP logits"):
        train_causal_lm_distill(teacher, student, cfg, tmp_path / "mtp_guard", batches)


def test_generic_distill_offline_full_logits_cache_uses_exact_entries(tmp_path: Path):
    teacher = RaisingTeacher()
    student = DummyHfMoeForCausalLM()
    cfg_base = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        calibration={"sample_count": 2, "sequence_length": 8},
        training={"train_steps": 1, "global_batch_size": 1, "micro_batch_size": 1, "warmup_steps": 0},
    )
    batches, _ = sample_calibration_tokens(cfg_base.calibration, vocab_size=student.config.vocab_size)
    logits = batches[0].new_zeros((1, 8, student.config.vocab_size)).float()
    logits[..., 0] = 2.0
    cache_path = tmp_path / "full_logits_cache.pt"
    write_full_logits_cache(cache_path, [(batches[0], logits)])
    cfg = cfg_base.model_copy(
        update={
            "kd": cfg_base.kd.model_copy(
                update={
                    "teacher_mode": "offline_full_logits_cache",
                    "offline_full_logits_cache_path": str(cache_path),
                    "mtp": cfg_base.kd.mtp.model_copy(update={"enabled": False}),
                }
            )
        }
    )

    result = train_causal_lm_distill(teacher, student, cfg, tmp_path / "offline_training", batches)

    assert result["global_step"] == 1
    assert result["logs"][0]["loss_kd"] > 0


def test_offline_full_logits_cache_uses_restricted_torch_loader(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "full_logits_cache.pt"
    input_ids = torch.tensor([[1, 2, 3]])
    logits = torch.zeros(1, 3, 5)
    write_full_logits_cache(cache_path, [(input_ids, logits)])
    real_load = torch.load
    captured = {}

    def recording_load(*args, **kwargs):
        captured["weights_only"] = kwargs.get("weights_only")
        return real_load(*args, **kwargs)

    monkeypatch.setattr("slimder_man.distill.offline_cache.torch.load", recording_load)

    OfflineFullLogitsCache.from_path(cache_path)

    assert captured["weights_only"] is True


def test_offline_topk_logits_cache_uses_restricted_torch_loader(monkeypatch, tmp_path: Path):
    cache_path = tmp_path / "topk_logits_cache.pt"
    input_ids = torch.tensor([[1, 2, 3]])
    logits = torch.zeros(1, 3, 5)
    logits[..., 2] = 4.0
    write_topk_logits_cache(cache_path, [(input_ids, logits)], top_k=2)
    real_load = torch.load
    captured = {}

    def recording_load(*args, **kwargs):
        captured["weights_only"] = kwargs.get("weights_only")
        return real_load(*args, **kwargs)

    monkeypatch.setattr("slimder_man.distill.offline_cache.torch.load", recording_load)

    out = OfflineTopKLogitsCache.from_path(cache_path).teacher_output(input_ids)

    assert captured["weights_only"] is True
    assert out.logits.shape == (1, 3, 5)
    assert out.logits.argmax(dim=-1).tolist() == [[2, 2, 2]]


def test_offline_topk_logits_cache_rejects_malformed_payloads(tmp_path: Path):
    input_ids = torch.tensor([[1, 2, 3]])
    base_entry = {
        "key": full_logits_cache_key(input_ids),
        "input_ids": input_ids,
        "topk_indices": torch.zeros(1, 3, 1, dtype=torch.long),
        "topk_values": torch.zeros(1, 3, 1),
    }

    cases = [
        ({"format": "slimder_topk_logits_cache_v1", "vocab_size": 5, "fill_value": float("nan"), "entries": [base_entry]}, "fill_value"),
        (
            {
                "format": "slimder_topk_logits_cache_v1",
                "vocab_size": 5,
                "entries": [{**base_entry, "topk_indices": torch.zeros(1, 3, 0, dtype=torch.long), "topk_values": torch.zeros(1, 3, 0)}],
            },
            "at least one top-k value",
        ),
        (
            {
                "format": "slimder_topk_logits_cache_v1",
                "vocab_size": 5,
                "entries": [{**base_entry, "topk_values": torch.full((1, 3, 1), float("inf"))}],
            },
            "non-finite values",
        ),
        ({"format": "slimder_topk_logits_cache_v1", "vocab_size": 5, "entries": [base_entry, base_entry]}, "duplicate entry"),
    ]
    for idx, (payload, message) in enumerate(cases):
        path = tmp_path / f"bad_topk_{idx}.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match=message):
            OfflineTopKLogitsCache.from_path(path)


def test_offline_full_logits_cache_requires_path(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = DummyHfMoeForCausalLM()
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        kd={"teacher_mode": "offline_full_logits_cache"},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    with pytest.raises(ValueError, match="offline_full_logits_cache_path"):
        train_causal_lm_distill(teacher, student, cfg, tmp_path / "offline_missing", batches)


def test_offline_topk_logits_cache_requires_path(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = DummyHfMoeForCausalLM()
    cfg = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        kd={"teacher_mode": "offline_topk_logit_cache"},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    with pytest.raises(ValueError, match="offline_topk_logits_cache_path"):
        train_causal_lm_distill(teacher, student, cfg, tmp_path / "topk_missing", batches)


def test_generic_distill_augmented_topk_cache_uses_approximate_logits_without_local_teacher(tmp_path: Path):
    teacher = RaisingTeacher()
    student = DummyHfMoeForCausalLM()
    cfg_base = SlimderConfig(
        project={"paper_faithful": False, "output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        calibration={"sample_count": 2, "sequence_length": 8},
        training={"train_steps": 1, "global_batch_size": 1, "micro_batch_size": 1, "warmup_steps": 0},
    )
    batches, _ = sample_calibration_tokens(cfg_base.calibration, vocab_size=student.config.vocab_size)
    logits = batches[0].new_full((1, 8, student.config.vocab_size), -5.0).float()
    logits[..., 3] = 5.0
    cache_path = tmp_path / "topk_logits_cache.pt"
    write_topk_logits_cache(cache_path, [(batches[0], logits)], top_k=4)
    cfg = cfg_base.model_copy(
        update={
            "kd": cfg_base.kd.model_copy(
                update={
                    "teacher_mode": "offline_topk_logit_cache",
                    "offline_topk_logits_cache_path": str(cache_path),
                    "mtp": cfg_base.kd.mtp.model_copy(update={"enabled": False}),
                }
            )
        }
    )

    result = train_causal_lm_distill(teacher, student, cfg, tmp_path / "topk_training", batches)

    assert result["global_step"] == 1
    assert result["logs"][0]["loss_kd"] > 0


def test_generic_distill_remote_worker_logits_mode_uses_client_not_local_teacher(tmp_path: Path):
    teacher = RaisingTeacher()
    student = DummyHfMoeForCausalLM()
    client = FakeWorkerLogitsClient(student.config.vocab_size)
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        calibration={"sample_count": 2, "sequence_length": 8},
        training={"train_steps": 1, "global_batch_size": 1, "micro_batch_size": 1, "warmup_steps": 0},
        kd={"teacher_mode": "remote_worker_full_logits", "mtp": {"enabled": False}},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    result = train_causal_lm_distill(
        teacher,
        student,
        cfg,
        tmp_path / "remote_worker_training",
        batches,
        teacher_logits_client=client,
    )

    assert result["global_step"] == 1
    assert client.calls == [(1, 8)]
    assert result["logs"][0]["loss_kd"] > 0


def test_remote_worker_mode_requires_api_url_without_injected_client(tmp_path: Path):
    teacher = DummyHfMoeForCausalLM()
    student = DummyHfMoeForCausalLM()
    cfg = SlimderConfig(
        project={"output_dir": str(tmp_path)},
        teacher={"load_mode": "transformers", "model_id_or_path": "dummy-hf-moe"},
        kd={"teacher_mode": "remote_worker_full_logits"},
    )
    batches, _ = sample_calibration_tokens(cfg.calibration, vocab_size=student.config.vocab_size)

    with pytest.raises(ValueError, match="runtime.worker.api_url"):
        train_causal_lm_distill(teacher, student, cfg, tmp_path / "remote_missing", batches)


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

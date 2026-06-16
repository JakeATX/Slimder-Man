from pathlib import Path

import json

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.train_loop import train_tiny_distill


def test_checkpoint_resume(tmp_path: Path):
    cfg = SlimderConfig(project={"output_dir": str(tmp_path)}, training={"train_steps": 3, "warmup_steps": 1})
    teacher = TinyMoEForCausalLM()
    student = TinyMoEForCausalLM()
    first = train_tiny_distill(teacher, student, cfg, tmp_path / "train")
    assert first["global_step"] == 3
    cfg2 = cfg.model_copy(deep=True)
    cfg2.training.train_steps = 5
    second = train_tiny_distill(teacher, student, cfg2, tmp_path / "train", resume=True)
    assert second["global_step"] == 5
    state = json.loads((tmp_path / "train" / "trainer_state.json").read_text())
    assert state["global_step"] == 5

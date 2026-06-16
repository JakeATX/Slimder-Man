from pathlib import Path

from slimder_man.adapters.tiny import TinyMoEForCausalLM
from slimder_man.analyze.architecture import describe_model
from slimder_man.calibration.collectors import collect_tiny_calibration
from slimder_man.calibration.datasets import sample_calibration_tokens
from slimder_man.compression.apply import compress_tiny_model
from slimder_man.config.schema import SlimderConfig
from slimder_man.distill.train_loop import train_tiny_distill
from slimder_man.eval.perplexity import tiny_perplexity


def test_direct_api_end_to_end(tmp_path: Path):
    cfg = SlimderConfig(project={"output_dir": str(tmp_path)})
    teacher = TinyMoEForCausalLM()
    arch = describe_model(teacher)
    assert arch["num_layers"] == 4
    batches, _ = sample_calibration_tokens(cfg.calibration)
    cal = collect_tiny_calibration(teacher, batches)
    student, manifest = compress_tiny_model(teacher, cfg, cal, tmp_path / "compressed")
    assert manifest["target"]["hidden_size"] == 12
    train = train_tiny_distill(teacher, student, cfg, tmp_path / "training")
    assert train["global_step"] == cfg.training.train_steps
    assert tiny_perplexity(student, batches[:8]) > 0
    generated = student.generate(batches[0][:, :2], max_new_tokens=8)
    assert generated.shape[1] == 10

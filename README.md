# Slimder Man

Slimder Man is a Python application for SlimQwen-style Mixture-of-Experts
compression and distillation. It provides strict `paper_faithful` behavior for
the SlimQwen recipe and explicit `augmented` features for practical deployment
workflows.

This v0.1 implementation provides a complete tiny-MoE CPU pipeline, a
HF-compatible dummy MoE pipeline for adapter-driven analyze/compress/distill/eval
coverage, and Qwen3-Next structural introspection scaffolding. Real
Qwen3-Next-80B local execution is exposed as explicit staged dry-run planning
until checkpoint-specific GPU validation is provisioned.

Quick local checks:

```bash
pip install -e .[dev]
slimder --help
slimqwen --help
pytest tests/unit -q
pytest tests/smoke -q -m "not gpu"
```

Tiny and HF-compatible smoke workflows:

```bash
slimder run src/slimder_man/config/examples/tiny_moe_cpu_smoke.yaml --json
slimder run src/slimder_man/config/examples/hf_dummy.yaml --json
slimder run src/slimder_man/config/examples/hf_dummy.yaml --dry-run --json
slimder launch src/slimder_man/config/examples/hf_dummy.yaml --backend local --json
```

The package targets Python `>=3.11,<3.13`.
GitHub Actions runs unit and non-GPU smoke tests on Python 3.11 and 3.12.

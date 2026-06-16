# Slimder Man

Slimder Man is a Python application for SlimQwen-style Mixture-of-Experts
compression and distillation. It provides strict `paper_faithful` behavior for
the SlimQwen recipe and explicit `augmented` features for practical deployment
workflows.

This v1 implementation provides a complete tiny-MoE end-to-end pipeline and
HF/Qwen3-Next adapter introspection scaffolding. Full Qwen3-Next structural
compression is intentionally gated in the CLI until a full checkpoint-specific
validation run is provisioned.

Quick smoke run:

```bash
pip install -e .[dev]
slimder --help
pytest tests/unit -q
pytest tests/smoke/test_cli_end_to_end_cpu.py -q
```

The package targets Python `>=3.11,<3.13`.

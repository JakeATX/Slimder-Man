# Slimder Man

Slimder Man is a Python application for SlimQwen-style Mixture-of-Experts
compression and distillation. It provides strict `paper_faithful` behavior for
the SlimQwen recipe and explicit `augmented` features for practical deployment
workflows.

This v0.1 implementation provides:

- complete tiny-MoE CPU analyze, recommend, compress, distill, eval, resume, UI,
  quantization, and checkpoint-validation smoke paths;
- an HF-compatible dummy MoE pipeline that exercises the adapter-driven
  analyze/compress/distill/eval flow and writes reloadable HF/safetensors
  checkpoints, manifests, tokenizer files, calibration artifacts, and hashes;
- structural Qwen3-Next and generic HF MoE adapters with fixture coverage for
  MoE detection, sparse layer indices, depth pruning, width slicing, expert
  pruning/merge, router transforms, and finite forward validation;
- local, SSH, SkyPilot, and worker-runner surfaces with dry-run plans, executable
  runner APIs, log streaming, stop/sync operations, and secret redaction.

Real Qwen3-Next-80B work remains hardware-gated: local full-model execution
requires explicit `runtime.local.allow_full_model_run=true`, and default CI uses
synthetic/tiny/HF-dummy fixtures rather than downloading the real 80B checkpoint.

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

Stage and artifact commands:

```bash
slimder analyze src/slimder_man/config/examples/hf_dummy.yaml --json
slimder recommend --config src/slimder_man/config/examples/hf_dummy.yaml --preset balanced_50 --json
slimder compress src/slimder_man/config/examples/hf_dummy.yaml --stage 1 --json
slimder distill src/slimder_man/config/examples/hf_dummy.yaml --stage 1 --json
slimder eval --checkpoint runs/hf_dummy_moe_smoke/training/final --json
slimder validate-checkpoint --checkpoint runs/hf_dummy_moe_smoke/checkpoints/stage_1_compressed --json
slimder quantize --config src/slimder_man/config/examples/hf_dummy.yaml --checkpoint runs/hf_dummy_moe_smoke/checkpoints/stage_1_compressed --json
```

Remote launch planning:

```bash
slimder launch src/slimder_man/config/examples/hf_dummy.yaml --backend ssh --json
slimder launch src/slimder_man/config/examples/hf_dummy.yaml --backend skypilot --json
slimder launch path/to/worker_config.yaml --backend worker --json
slimder worker --json
slimder worker --host 0.0.0.0 --auth-token change-me
```

For SSH/SkyPilot configs, `runtime.ssh.dry_run=false` or
`runtime.skypilot.dry_run=false` switches `slimder launch` from plan generation
to the executable runner path.
For worker configs, set `runtime.worker.api_url` to a running `slimder worker`
service; launch submits the YAML as job input so the worker does not need a
shared local config path. Worker defaults to `127.0.0.1`; non-local binds
require `--auth-token` or `SLIMDER_WORKER_TOKEN` because `/v1/jobs` executes
local subprocesses.

The package targets Python `>=3.11,<3.13`.
GitHub Actions runs unit and non-GPU smoke tests on Python 3.11 and 3.12.

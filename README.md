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
The same guardrails apply to Qwen3.6-35B-A3B: config-only analysis and
recommendation are safe locally, while real compression/distillation should use
SSH, SkyPilot, or a Worker API endpoint with high-memory GPUs and full-logit
teacher access.

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
slimder quantize src/slimder_man/config/examples/hf_dummy.yaml runs/hf_dummy_moe_smoke/checkpoints/stage_1_compressed --json
```

The v0.1 quantize command is an augmented fake-quant/export smoke path: it
writes dequantized tensors plus manifests, not packed production quantized MoE
kernels.

Remote launch planning:

```bash
slimder launch src/slimder_man/config/examples/hf_dummy.yaml --backend ssh --json
slimder launch src/slimder_man/config/examples/hf_dummy.yaml --backend skypilot --json
slimder launch path/to/worker_config.yaml --backend worker --json
slimder worker --json
slimder worker --host 0.0.0.0 --auth-token change-me
slimder worker-preflight --config path/to/worker_config.yaml --json
slimder worker-status --config path/to/worker_config.yaml --job-id JOB_ID --json
slimder worker-logs --config path/to/worker_config.yaml --job-id JOB_ID --json
slimder worker-artifacts --config path/to/worker_config.yaml --job-id JOB_ID --json
slimder worker-sync --config path/to/worker_config.yaml --job-id JOB_ID --out runs/worker_copy --json
slimder worker-stop --config path/to/worker_config.yaml --job-id JOB_ID --json
```

Qwen3.6-35B-A3B planning:

```bash
slimder compute-guidance src/slimder_man/config/examples/qwen36_35b_a3b_remote.yaml --json
slimder analyze src/slimder_man/config/examples/qwen36_35b_a3b_remote.yaml --config-only --json
slimder recommend --config src/slimder_man/config/examples/qwen36_35b_a3b_remote.yaml --preset balanced_50 --write-config src/slimder_man/config/examples/qwen36_35b_a3b_remote.yaml --config-only --json
slimder run src/slimder_man/config/examples/qwen36_35b_a3b_remote.yaml --dry-run --json
slimder launch src/slimder_man/config/examples/qwen36_35b_a3b_remote.yaml --backend skypilot --json
```

For Qwen3.6-35B-A3B, Slimder uses a known profile of 35B total parameters,
roughly 3B active parameters, 40 layers, hidden size 2048, 256 routed experts,
8 routed experts active per token, and 1 shared expert. The compute guidance
reports the bf16/fp16 teacher-weight floor, teacher+student training floor, and
whether API use is compatible with paper-faithful KD. A generic chat-completions
API is not enough for paper-faithful distillation unless the service also
returns exact full-vocabulary logits; use online remote logits, an exact
full-logit cache, or `remote_worker_full_logits`.

For SSH/SkyPilot configs, `runtime.ssh.dry_run=false` or
`runtime.skypilot.dry_run=false` switches `slimder launch` from plan generation
to the executable runner path.
For worker configs, set `runtime.worker.api_url` to a running `slimder worker`
service; launch submits the YAML as job input so the worker does not need a
shared local config path. Worker defaults to `127.0.0.1`; non-local binds
require `--auth-token` or `SLIMDER_WORKER_TOKEN` because `/v1/jobs` executes
local subprocesses. Treat the worker API as trusted infrastructure; artifact
sync packages the job's recorded artifact paths for retrieval.

The package targets Python `>=3.11,<3.13` and supports Transformers
`>=4.55,<6`, including the packed/fused Transformers 5.x Qwen MoE layout.
GitHub Actions runs unit and non-GPU smoke tests on Python 3.11 and 3.12.

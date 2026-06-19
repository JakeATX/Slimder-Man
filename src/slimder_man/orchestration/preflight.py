from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreflightProbe:
    name: str
    command: str
    actionable_failure: str


def ssh_preflight_probes(remote_root: str) -> list[PreflightProbe]:
    return [
        PreflightProbe(
            "python",
            f"cd {remote_root} && python --version && python -m pip --version",
            "Install Python and pip on the remote host, then rerun launch.",
        ),
        PreflightProbe(
            "cuda",
            f"cd {remote_root} && command -v nvidia-smi >/dev/null && nvidia-smi",
            "CUDA is not visible. Check NVIDIA drivers, container runtime, or choose a CPU/tiny run.",
        ),
        PreflightProbe(
            "disk",
            (
                f"cd {remote_root} && df -h . && "
                "python -c \"import shutil,sys; free=shutil.disk_usage('.').free; "
                "print(str(free // (1024**3)) + 'GB free'); sys.exit(0 if free >= 20 * 1024**3 else 1)\""
            ),
            "Free remote disk space or configure a larger volume before compression/training.",
        ),
        PreflightProbe(
            "torch",
            f"cd {remote_root} && python -c \"import torch; print(torch.__version__)\"",
            "Install the project with dev dependencies or a CUDA-compatible PyTorch wheel.",
        ),
    ]

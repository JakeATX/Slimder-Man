from __future__ import annotations

import yaml

from slimder_man.config.schema import SlimderConfig
from slimder_man.utils.hashing import redact_secret


def skypilot_yaml(cfg: SlimderConfig) -> str:
    data = {
        "name": cfg.runtime.skypilot.cluster_name,
        "resources": {"accelerators": cfg.runtime.skypilot.accelerators, "cloud": cfg.runtime.skypilot.cloud},
        "workdir": ".",
        "setup": "pip install -e .[dev]",
        "run": "slimder run --config config.yaml",
    }
    return redact_secret(yaml.safe_dump(data, sort_keys=False))

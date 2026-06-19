from __future__ import annotations

import argparse
import json
from pathlib import Path

from slimder_man.config.schema import load_config, save_config


def materialize_remote_config(source: str | Path, destination: str | Path, output_dir: str) -> dict[str, str]:
    cfg = load_config(source)
    cfg.project.output_dir = output_dir
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, destination)
    return {"source": str(source), "destination": str(destination), "output_dir": output_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize a remote-safe Slimder config.")
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = materialize_remote_config(args.source, args.destination, args.output_dir)
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"wrote {payload['destination']}")


if __name__ == "__main__":
    main()

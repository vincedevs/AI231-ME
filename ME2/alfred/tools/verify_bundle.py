#!/usr/bin/env python3
"""Verify Alfred's configuration, model hashes, and feedback assets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alfred.config import load_config
from alfred.health import verify_install


def main() -> int:
    config = load_config()
    report = verify_install(
        config,
        require_vcm=config.document["vcm"]["mode"] == "onnx",
        run_models=True,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

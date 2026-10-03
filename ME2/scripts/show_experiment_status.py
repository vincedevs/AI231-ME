#!/usr/bin/env python3
"""Show or continuously watch VCM experiment progress."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models/experiments/ai231_me2_voice_commands",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    return parser.parse_args()


def read_status(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def experiment_status_paths(root: Path) -> list[Path]:
    return sorted((root / "runs").rglob("seed_*/status.json"))


def render(root: Path) -> str:
    suite = read_status(root / "status.json")
    lines = [f"Experiment root: {root}"]
    if not suite:
        return "\n".join([*lines, "No experiment suite has started."])
    lines.append(
        f"Study: {suite.get('state', 'unknown')} | "
        f"run {suite.get('experiment', '-')}/{suite.get('experiments', '-')} | "
        f"elapsed {float(suite.get('elapsed_seconds', 0)) / 3600:.2f} h"
    )
    current = suite.get("current")
    if current:
        lines.append(
            "Current: "
            + " | ".join(
                str(current.get(key, ""))
                for key in ("architecture", "data_condition", "rejection_strategy")
            )
        )
    statuses = experiment_status_paths(root)
    for path in statuses:
        status = read_status(path)
        identity = " | ".join(
            str(status.get(key, ""))
            for key in ("architecture", "data_condition", "rejection_strategy")
        )
        identity += f" | seed {status.get('seed', '')}"
        progress = status.get("state", "unknown")
        if progress == "training":
            progress += (
                f" epoch {status.get('epoch')}/{status.get('epochs')}"
                f" batch {status.get('batch')}/{status.get('batches')}"
                f" loss {float(status.get('mean_loss', 0)):.4f}"
            )
        lines.append(f"- {identity}: {progress}")
    for path in sorted((root / "optimization").glob("*/trials.csv")):
        trials = max(0, len(path.read_text(encoding="utf-8").splitlines()) - 1)
        complete = (path.parent / "best_hyperparameters.json").is_file()
        lines.append(
            f"- HPO {path.parent.name}: {trials} recorded trials"
            + ("; best parameters selected" if complete else "")
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = args.output.resolve()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        print(render(root), flush=True)
        if not args.watch:
            return 0
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())

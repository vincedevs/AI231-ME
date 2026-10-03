#!/usr/bin/env python3
"""Build or validate the reproducible VCM dataset from the frozen raw release."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset_pipeline.prepare import prepare_dataset, resolve, validate_prepared_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/dataset.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--without-supplements",
        action="store_true",
        help="Prepare only the frozen Hugging Face release; omit approved previous real audio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = resolve(PROJECT_ROOT, config["output_root"])
    if args.validate_only:
        report = validate_prepared_dataset(output)
        print(json.dumps(report, indent=2))
        return 0
    release = prepare_dataset(
        PROJECT_ROOT,
        config,
        overwrite=args.overwrite,
        include_supplements=not args.without_supplements,
    )
    print(json.dumps(release["summary"], indent=2))
    print(f"Prepared dataset: {output}")
    print("Holdout remains sealed and is not consumed by the training script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the reproducible VCM model-selection study on one accelerator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def comma_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments.json")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--architectures", type=comma_list)
    parser.add_argument("--data-conditions", type=comma_list)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select(requested: list[str] | None, available: list[str], name: str) -> list[str]:
    if requested is None:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown {name}: {unknown}; available: {available}")
    return requested


def restrict_to_one_cuda_gpu(index: int) -> str:
    """Expose exactly one GPU, preserving the scheduler's assigned ordering."""
    if index < 0:
        raise ValueError("--gpu-index must be non-negative")
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = assigned.split(",", 1)[0].strip() if assigned else str(index)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return visible


def choose_device(requested: str):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access it")
        return torch.device("cuda:0")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but PyTorch cannot access it")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe(config: dict[str, Any], architectures: list[str], conditions: list[str]) -> None:
    if config.get("study_mode") == "fixed_condition_ablation":
        print("VCM fixed-condition ablation")
        print(f"  Architecture: {', '.join(architectures)}")
        print(f"  Data conditions: {', '.join(conditions)}")
        print(f"  Seeds: {', '.join(str(seed) for seed in config['replication_seeds'])}")
        print("  HPO: disabled; prior-selected Tiny Conformer hyperparameters are fixed")
        print("  Selection: validation aggregates only; test metrics are report-only")
        print("  Final holdout: sealed; this script never reads it")
        return
    tuning = config["tuning"]
    comparison = config["comparison"]
    print("VCM study plan")
    print(f"  Architectures: {', '.join(architectures)}")
    print(f"  Data conditions: {', '.join(conditions)}")
    print(
        "  HPO: "
        f"{tuning['trials_per_architecture']} trials/architecture on "
        f"{tuning['reference_data_condition']} + {tuning['reference_rejection_strategy']}"
    )
    print(f"  Controlled architecture/data runs: {len(architectures) * len(conditions)}")
    print(f"  Rejection ablation finalists: {comparison['rejection_finalists']}")
    print(
        f"  Replication finalists: {comparison['replication_finalists']} × "
        f"{len(config['replication_seeds'])} seeds"
    )
    print("  Final holdout: sealed; this script never reads it")


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    architectures = select(args.architectures, list(config["architectures"]), "architectures")
    conditions = select(args.data_conditions, list(config["data_conditions"]), "data conditions")
    if args.smoke_test:
        config = json.loads(json.dumps(config))
        # Keep preflight artifacts beside, not inside, the full-study output.
        # This preserves the clean-root invariant enforced by the study guard.
        config["paths"]["output_root"] += "_smoke"
    describe(config, architectures, conditions)
    if args.dry_run:
        return 0

    visible_gpu = restrict_to_one_cuda_gpu(args.gpu_index)
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "models/.matplotlib-cache"))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    from vcm_training.experiment import run_study

    device = choose_device(args.device)
    if device.type == "cuda" and torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"One-GPU policy violated: PyTorch sees {torch.cuda.device_count()} GPUs"
        )
    print(f"Device: {device}; CUDA_VISIBLE_DEVICES={visible_gpu!r}", flush=True)
    selection = run_study(
        PROJECT_ROOT, config, device, architectures, conditions, args.smoke_test
    )
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

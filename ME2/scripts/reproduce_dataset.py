#!/usr/bin/env python3
"""Download, prepare, and validate the dataset through one auditable command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Reproduce only the pinned Hugging Face benchmark without local supplements.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use already downloaded source files after checksum verification.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def supplement_manifest_paths(config: dict) -> list[Path]:
    previous = config["previous_dataset"]
    root = PROJECT_ROOT / previous["root"]
    return [
        root / name
        for name in previous["supported_manifests"] + previous["unsupported_manifests"]
    ]


def require_supplement_source(config: dict) -> None:
    required = supplement_manifest_paths(config)
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        release = PROJECT_ROOT / "dataset" / "raw" / "alfred_vcm_supplement"
        if release.is_dir():
            run(
                [
                    sys.executable,
                    "scripts/install_supplement_release.py",
                    "--release",
                    str(release),
                ]
            )
            missing = [
                str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()
            ]
        if missing:
            raise FileNotFoundError(
                "The exact supplemental source is not installed. Missing: "
                + ", ".join(missing)
                + ". Use --benchmark-only to reproduce the pinned benchmark without it."
            )


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if not args.skip_download:
        download_command = [
            sys.executable,
            "scripts/download_dataset.py",
            "--config",
            str(config_path),
        ]
        supplement_is_installed = all(
            path.is_file() for path in supplement_manifest_paths(config)
        )
        if not args.benchmark_only and not supplement_is_installed:
            download_command.append("--with-supplement-release")
        run(download_command)

    if not args.benchmark_only:
        require_supplement_source(config)

    preparation_config = config_path
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if args.benchmark_only:
        benchmark_config = dict(config)
        benchmark_config["output_root"] = "dataset/prepared_benchmark_only"
        benchmark_config["supplementation"] = {
            **benchmark_config["supplementation"],
            "enabled": False,
        }
        benchmark_config["supplemental_synthetic"] = {
            **benchmark_config["supplemental_synthetic"],
            "enabled": False,
        }
        temporary_directory = tempfile.TemporaryDirectory(prefix="me2-benchmark-config-")
        preparation_config = Path(temporary_directory.name) / "dataset.json"
        preparation_config.write_text(
            json.dumps(benchmark_config, indent=2) + "\n", encoding="utf-8"
        )

    try:
        command = [
            sys.executable,
            "scripts/prepare_dataset.py",
            "--config",
            str(preparation_config),
            "--overwrite",
        ]
        if args.benchmark_only:
            command.append("--without-supplements")
        run(command)
        run(
            [
                sys.executable,
                "scripts/prepare_dataset.py",
                "--config",
                str(preparation_config),
                "--validate-only",
            ]
        )
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

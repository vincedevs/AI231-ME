#!/usr/bin/env python3
"""Download and checksum-verify the pinned inputs for dataset preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "dataset.json"
PRIMARY_REPOSITORY = "airimonda/ai231-me2-voice-commands"
SUPPLEMENT_REPOSITORY = "vincedevs/alfred-vcm-supplement"
SUPPLEMENT_REVISION = "c376ada37c0eea13a4db1893cc854a347e5d25b5"


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_primary_download(config: dict) -> None:
    source_root = resolve(config["source_root"])
    failures = []
    for relative, expected in config["source_shards_sha256"].items():
        path = source_root / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"checksum mismatch {relative}: {actual}")
    if failures:
        raise ValueError("Pinned Hugging Face download failed verification: " + "; ".join(failures))


def download_primary(config: dict) -> Path:
    source_root = resolve(config["source_root"])
    patterns = sorted(config["source_shards_sha256"])
    patterns.extend(["README.md", "LICENSE*", ".gitattributes", "variations.csv"])
    print(
        f"Downloading {PRIMARY_REPOSITORY}@{config['source_revision']} "
        f"to {source_root}"
    )
    snapshot_download(
        repo_id=PRIMARY_REPOSITORY,
        repo_type="dataset",
        revision=str(config["source_revision"]),
        local_dir=source_root,
        allow_patterns=patterns,
    )
    verify_primary_download(config)
    print("Pinned primary dataset verified.")
    return source_root


def download_supplement(repository: str, revision: str, destination: Path) -> Path:
    print(f"Downloading {repository}@{revision} to {destination}")
    snapshot_download(
        repo_id=repository,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
    )
    summary_path = destination / "export_summary.json"
    if not summary_path.is_file():
        raise ValueError(
            "Supplement release is incomplete: export_summary.json was not downloaded"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {"train": 15037, "validation": 1413}
    if summary.get("records") != 16450 or summary.get("splits") != expected:
        raise ValueError(f"Unexpected supplement release summary: {summary}")
    print("Supplement membership and split counts verified.")
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--with-supplement-release",
        action="store_true",
        help="Also download the private supplemental provenance release.",
    )
    parser.add_argument("--supplement-repository", default=SUPPLEMENT_REPOSITORY)
    parser.add_argument("--supplement-revision", default=SUPPLEMENT_REVISION)
    parser.add_argument(
        "--supplement-destination",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "raw" / "alfred_vcm_supplement",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    download_primary(config)
    if args.with_supplement_release:
        download_supplement(
            args.supplement_repository,
            args.supplement_revision,
            args.supplement_destination.resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

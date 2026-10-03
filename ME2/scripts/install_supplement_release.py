#!/usr/bin/env python3
"""Install a complete downloaded supplemental release as a preparation input."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = PROJECT_ROOT / "dataset" / "raw" / "alfred_vcm_supplement"
DEFAULT_DESTINATION = PROJECT_ROOT / "dataset" / "training"
EXPECTED_COUNTS = {"train": 15037, "validation": 1413}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def required_audio(release: Path, records: list[dict]) -> list[Path]:
    missing = []
    for record in records:
        audio_path = record.get("audio_path")
        if not audio_path or not (release / audio_path).is_file():
            missing.append(Path(str(record.get("prepared_audio_path") or record["sample_id"])))
    return missing


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    release = args.release.resolve()
    destination = args.destination.resolve()

    records = []
    for split, expected in EXPECTED_COUNTS.items():
        path = release / "manifests" / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing supplemental manifest: {path}")
        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} {split} records, found {len(rows)}")
        records.extend(rows)

    missing = required_audio(release, records)
    if missing:
        raise FileNotFoundError(
            f"Supplement release has metadata for {len(records)} records but is missing audio "
            f"for {len(missing)} records. First missing item: {missing[0]}. Obtain restricted "
            "source audio under its original terms before installing the exact release."
        )

    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"Destination exists: {destination}; pass --overwrite")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="supplement-install-", dir=destination.parent) as temp:
        temporary = Path(temp) / "training"
        supported = []
        unsupported = []
        copied_pcm = set()

        for record in records:
            source_audio = release / record["audio_path"]
            relative_audio = Path("audio") / source_audio.parent.name / source_audio.name
            if record["pcm_sha256"] not in copied_pcm:
                link_or_copy(source_audio, temporary / relative_audio)
                copied_pcm.add(record["pcm_sha256"])

            installed = dict(record)
            installed["audio_path"] = (Path("dataset") / "training" / relative_audio).as_posix()
            installed.pop("audio_included", None)
            installed.pop("redistribution_status", None)
            installed.pop("prepared_audio_path", None)
            if installed["supported"]:
                supported.append(installed)
            else:
                unsupported.append(installed)

        manifest_root = temporary / "manifests"
        supported_root = manifest_root / "speaker_generalization"
        unsupported_root = manifest_root / "unsupported"
        supported_root.mkdir(parents=True)
        unsupported_root.mkdir(parents=True)

        for root, selected in ((supported_root, supported), (unsupported_root, unsupported)):
            for split in ("train", "validation", "test"):
                path = root / f"{split}.jsonl"
                rows = selected if split == "train" else []
                with path.open("w", encoding="utf-8") as handle:
                    for row in sorted(rows, key=lambda item: str(item["sample_id"])):
                        handle.write(json.dumps(row, sort_keys=True) + "\n")

        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)

    print(
        json.dumps(
            {
                "destination": str(destination),
                "records": len(records),
                "supported": len(supported),
                "unsupported": len(unsupported),
                "audio_files": len(copied_pcm),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

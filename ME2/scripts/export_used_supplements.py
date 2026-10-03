#!/usr/bin/env python3
"""Export the exact supplemental records used by the selected VCM experiment.

The export always includes metadata for every supplemental real-audio record in
the prepared train and validation manifests. Audio is copied only for sources
whose redistribution terms permit the derived research copy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_ROOT = PROJECT_ROOT / "dataset" / "prepared"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "export" / "alfred-vcm-supplement"
EXPECTED_COUNTS = {"train": 15037, "validation": 1413}

# These are the only sources in the selected supplement whose current terms
# permit redistribution of this non-commercial derived copy with attribution.
REDISTRIBUTABLE_AUDIO_SOURCES = {"slurp", "esc50"}

SOURCE_STATUS = {
    "stop": "metadata_only_source_terms_prohibit_redistribution",
    "slurp": "audio_included_cc_by_nc_4_0",
    "fsc": "metadata_only_adapted_audio_may_not_be_shared",
    "snips": "metadata_only_transformed_copy_not_redistributed",
    "rochester": "metadata_only_authoritative_redistribution_terms_missing",
    "esc50": "audio_included_cc_by_nc_3_0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, default=DEFAULT_PREPARED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset_card(output_root: Path, source_counts: Counter[str]) -> None:
    counts = "\n".join(f"| {source} | {count:,} |" for source, count in sorted(source_counts.items()))
    card = f"""---
license: other
license_name: per-source-research-only
pretty_name: Alfred VCM Supplemental Real Audio
task_categories:
- audio-classification
language:
- en
---

# Alfred VCM supplemental records

This private research release records the exact **16,450 supplemental real-audio
records** used by the selected Alfred VCM experiment: 15,037 training records
and 1,413 validation records.

Every record retains its original source identity, split, speaker and recording
group, canonical annotation, and decoded-audio checksum. Audio is included only
when the source terms permit redistribution of this derived non-commercial copy.
Restricted records remain metadata-only and must be reacquired from their
original source under its terms.

| Source | Records |
| --- | ---: |
{counts}

## Important

- This is not a uniformly licensed dataset.
- It is for non-commercial research and education only.
- `manifests/train.jsonl` and `manifests/validation.jsonl` define exact membership.
- `source_inventory.csv` states whether audio is included for each source.
- Test and holdout records are not included.
- Synthetic supplemental and synthetic-negative records are not included.

See `LICENSES/source-terms.md` before use or redistribution.
"""
    (output_root / "README.md").write_text(card, encoding="utf-8")


def write_source_terms(output_root: Path) -> None:
    terms = """# Source terms

This export preserves source-level licensing rather than applying one licence to
the combined collection.

- SLURP audio: CC BY-NC 4.0. Included with attribution for non-commercial use.
  https://github.com/pswietojanski/slurp/blob/master/LICENSE.txt
- ESC-50: CC BY-NC 3.0. Included with source attribution for non-commercial use.
  https://github.com/karolpiczak/ESC-50/blob/master/LICENSE
- STOP: metadata only. The audio is not redistributed.
  https://dl.fbaipublicfiles.com/stop/LICENSE.txt
- Fluent Speech Commands: metadata only. The adapted audio is not redistributed.
  https://fluent.ai/wp-content/uploads/2021/04/Fluent_Speech_Commands_Public_License.pdf
- SNIPS SLU: metadata only. The transformed copy is not redistributed.
  https://github.com/sonos/spoken-language-understanding-research-datasets
- Rochester Smart Speaker Commands: metadata only because authoritative
  redistribution terms are not retained in this project.

The manifest's `audio_included` and `redistribution_status` fields are
authoritative for this export.
"""
    license_dir = output_root / "LICENSES"
    license_dir.mkdir(parents=True, exist_ok=True)
    (license_dir / "source-terms.md").write_text(terms, encoding="utf-8")


def main() -> int:
    args = parse_args()
    prepared_root = args.prepared_root.resolve()
    output_root = args.output_root.resolve()

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)

    (output_root / "manifests").mkdir(parents=True)
    source_counts: Counter[str] = Counter()
    included_audio: set[str] = set()
    exported_counts: dict[str, int] = {}

    for split, expected_count in EXPECTED_COUNTS.items():
        source_manifest = prepared_root / "manifests" / f"{split}.jsonl"
        destination_manifest = output_root / "manifests" / f"{split}.jsonl"
        count = 0

        with source_manifest.open(encoding="utf-8") as source, destination_manifest.open(
            "w", encoding="utf-8"
        ) as destination:
            for line in source:
                record = json.loads(line)
                if not record.get("is_supplemental"):
                    continue
                if record.get("is_supplemental_synthetic"):
                    continue

                source_name = record["source_dataset"]
                if source_name not in SOURCE_STATUS:
                    raise ValueError(f"Missing redistribution decision for source: {source_name}")

                source_counts[source_name] += 1
                count += 1
                prepared_audio_path = prepared_root / record["audio_path"]
                audio_included = source_name in REDISTRIBUTABLE_AUDIO_SOURCES

                exported = dict(record)
                exported["prepared_audio_path"] = record["audio_path"]
                exported["audio_included"] = audio_included
                exported["redistribution_status"] = SOURCE_STATUS[source_name]

                if audio_included:
                    relative_audio = Path(record["audio_path"])
                    exported["audio_path"] = relative_audio.as_posix()
                    destination_audio = output_root / relative_audio
                    if exported["pcm_sha256"] not in included_audio:
                        destination_audio.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            os.link(prepared_audio_path, destination_audio)
                        except OSError:
                            shutil.copy2(prepared_audio_path, destination_audio)
                        included_audio.add(exported["pcm_sha256"])
                else:
                    exported["audio_path"] = None

                destination.write(json.dumps(exported, sort_keys=True) + "\n")

        if count != expected_count:
            raise ValueError(f"Expected {expected_count} {split} records, found {count}")
        exported_counts[split] = count

    inventory_path = output_root / "source_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "records", "audio_included", "redistribution_status"])
        for source_name, count in sorted(source_counts.items()):
            writer.writerow(
                [
                    source_name,
                    count,
                    source_name in REDISTRIBUTABLE_AUDIO_SOURCES,
                    SOURCE_STATUS[source_name],
                ]
            )

    write_dataset_card(output_root, source_counts)
    write_source_terms(output_root)

    checksummed_files = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    with (output_root / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in checksummed_files:
            handle.write(f"{sha256(path)}  {path.relative_to(output_root).as_posix()}\n")

    summary = {
        "records": sum(exported_counts.values()),
        "splits": exported_counts,
        "source_counts": dict(sorted(source_counts.items())),
        "included_audio_files": len(included_audio),
        "restricted_records_are_metadata_only": True,
    }
    (output_root / "export_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Exported to: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

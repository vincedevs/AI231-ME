#!/usr/bin/env python3
"""Copy reviewed runtime models into Alfred and write a checksummed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

ALFRED_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def copy_artifact(source: Path, relative_destination: str) -> dict[str, Any]:
    destination = ALFRED_ROOT / relative_destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    return {
        "path": relative_destination,
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "source_name": source.name,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hey-alfred", type=Path, required=True)
    parser.add_argument("--im-batman", type=Path, required=True)
    parser.add_argument("--melspectrogram", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--silero-vad", type=Path, required=True)
    parser.add_argument("--vcm", type=Path)
    parser.add_argument("--vcm-metadata", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (args.vcm is None) != (args.vcm_metadata is None):
        raise ValueError("--vcm and --vcm-metadata must be supplied together")
    sources = {
        "hey_alfred": require_file(args.hey_alfred, "Hey Alfred model"),
        "im_batman": require_file(args.im_batman, "I'm Batman model"),
        "melspectrogram": require_file(args.melspectrogram, "openWakeWord Mel model"),
        "embedding": require_file(args.embedding, "openWakeWord embedding model"),
        "silero": require_file(args.silero_vad, "Silero VAD model"),
    }
    artifacts: dict[str, Any] = {
        "wake_word": {
            "files": {
                name: copy_artifact(sources[name], f"assets/models/wake_word/{name}.onnx")
                for name in ("hey_alfred", "im_batman", "melspectrogram", "embedding")
            }
        },
        "vad": {
            "files": {
                "silero": copy_artifact(sources["silero"], "assets/models/vad/silero_vad.onnx")
            }
        },
    }
    existing_manifest = ALFRED_ROOT / "assets" / "models" / "manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        packaged_tts = existing.get("artifacts", {}).get("tts")
        if isinstance(packaged_tts, dict):
            artifacts["tts"] = packaged_tts
    if args.vcm is not None and args.vcm_metadata is not None:
        model = copy_artifact(require_file(args.vcm, "VCM model"), "assets/models/vcm/model.onnx")
        metadata = copy_artifact(
            require_file(args.vcm_metadata, "VCM metadata"),
            "assets/models/vcm/metadata.json",
        )
        artifacts["vcm"] = {
            "path": model["path"],
            "sha256": model["sha256"],
            "size_bytes": model["size_bytes"],
            "metadata_path": metadata["path"],
            "metadata_sha256": metadata["sha256"],
        }

    manifest = {
        "bundle_version": "1.0.0",
        "target": "raspberry_pi_5_arm64",
        "sample_rate": 16000,
        "artifacts": artifacts,
    }
    destination = ALFRED_ROOT / "assets" / "models" / "manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"Wrote {destination}")
    for group, value in artifacts.items():
        count = len(value.get("files", {})) if "files" in value else 2
        print(f"- {group}: {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

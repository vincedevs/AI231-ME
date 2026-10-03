#!/usr/bin/env python3
"""Install Moonshine v2 Tiny English for Alfred's one-turn clarifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ALFRED_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27"
MODEL_URL = (
    f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_NAME}.tar.bz2"
)
DESTINATION = ALFRED_ROOT / "assets" / "models" / "asr" / "moonshine-v2-tiny-en"
REQUIRED_MEMBERS = {
    "encoder_model.ort": 1_000_000,
    "decoder_model_merged.ort": 1_000_000,
    "tokens.txt": 1_000,
}
OPTIONAL_MEMBERS = ("LICENSE", "README.md")


class ModelDownloadError(RuntimeError):
    """The remote model archive could not be downloaded safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def download_archive(destination: Path, attempts: int = 3) -> None:
    if attempts <= 0:
        raise ValueError("Download attempts must be positive")
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Alfred-model-installer"})
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        destination.unlink(missing_ok=True)
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as response,
                destination.open("wb") as output,
            ):
                total = int(response.headers.get("Content-Length", 0))
                received = 0
                while block := response.read(1024 * 1024):
                    output.write(block)
                    received += len(block)
                    progress = (
                        f"{received / 1048576:.1f}/{total / 1048576:.1f} MiB "
                        f"({received / total:.0%})"
                        if total
                        else f"{received / 1048576:.1f} MiB"
                    )
                    print(
                        f"\rDownloading Moonshine v2 Tiny English: {progress}",
                        end="",
                        flush=True,
                    )
            print()
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = error
            destination.unlink(missing_ok=True)
            if attempt < attempts:
                delay = 2 ** (attempt - 1)
                print(
                    f"\nDownload attempt {attempt}/{attempts} failed; "
                    f"retrying in {delay} second{'s' if delay != 1 else ''}...",
                    file=sys.stderr,
                )
                time.sleep(delay)
    reason = getattr(last_error, "reason", last_error)
    raise ModelDownloadError(
        f"Unable to download Moonshine after {attempts} attempts: {reason}. "
        "Check that github.com resolves on this network, then retry. "
        "Alternatively, download the official archive in a browser and pass "
        "its path with --archive."
    ) from last_error


def find_member(archive: tarfile.TarFile, filename: str) -> tarfile.TarInfo | None:
    matches = [
        member
        for member in archive.getmembers()
        if member.isfile() and Path(member.name).name == filename
    ]
    if len(matches) > 1:
        raise ValueError(f"Archive contains multiple files named {filename}")
    return matches[0] if matches else None


def install_archive(archive_path: Path, destination: Path, overwrite: bool) -> dict:
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Model is already installed: {destination}. Use --overwrite.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".moonshine-v2-", dir=destination.parent))
    try:
        installed: dict[str, dict[str, object]] = {}
        with tarfile.open(archive_path, mode="r:*") as archive:
            for filename, minimum_size in REQUIRED_MEMBERS.items():
                member = find_member(archive, filename)
                if member is None or member.size < minimum_size:
                    raise ValueError(f"Archive is missing a valid {filename}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"Unable to read {filename} from the archive")
                output = temporary / filename
                with source, output.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
                installed[filename] = {
                    "size_bytes": output.stat().st_size,
                    "sha256": sha256_file(output),
                }
            for filename in OPTIONAL_MEMBERS:
                member = find_member(archive, filename)
                if member is None:
                    continue
                source = archive.extractfile(member)
                if source is not None:
                    with source, (temporary / filename).open("wb") as stream:
                        shutil.copyfileobj(source, stream)

        installation = {
            "engine": "sherpa_onnx_moonshine_v2",
            "model": "Moonshine v2 Tiny English quantized",
            "source_url": MODEL_URL,
            "archive_sha256": sha256_file(archive_path),
            "installed_at_utc": datetime.now(UTC).isoformat(),
            "files": installed,
        }
        (temporary / "installation.json").write_text(
            json.dumps(installation, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return installation
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        help="Install an already downloaded archive instead of downloading it",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="alfred-moonshine-v2-") as temporary_directory:
        if args.archive is None:
            archive_path = Path(temporary_directory) / f"{MODEL_NAME}.tar.bz2"
            try:
                download_archive(archive_path)
            except ModelDownloadError as error:
                print(f"error: {error}", file=sys.stderr)
                print(f"source: {MODEL_URL}", file=sys.stderr)
                return 2
        else:
            archive_path = args.archive.expanduser().resolve()
            if not archive_path.is_file():
                raise FileNotFoundError(f"Archive does not exist: {archive_path}")
        installation = install_archive(archive_path, DESTINATION, args.overwrite)

    print(f"Installed Moonshine clarification ASR at {DESTINATION}")
    print(f"Archive SHA-256: {installation['archive_sha256']}")
    for filename, metadata in installation["files"].items():
        print(f"- {filename}: {metadata['size_bytes']} bytes, SHA-256 {metadata['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

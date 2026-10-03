from __future__ import annotations

import io
import json
import tarfile
import urllib.error

import pytest

from tools import install_clarification_asr


def add_tar_file(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


def test_installer_extracts_only_required_files_and_records_hashes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        install_clarification_asr,
        "REQUIRED_MEMBERS",
        {"encoder_model.ort": 4, "decoder_model_merged.ort": 4, "tokens.txt": 4},
    )
    archive_path = tmp_path / "model.tar.bz2"
    with tarfile.open(archive_path, "w:bz2") as archive:
        add_tar_file(archive, "model/encoder_model.ort", b"encoder-bytes")
        add_tar_file(archive, "model/decoder_model_merged.ort", b"decoder-bytes")
        add_tar_file(archive, "model/tokens.txt", b"token-bytes")
        add_tar_file(archive, "model/unused.ort", b"not-installed")

    destination = tmp_path / "installed"
    metadata = install_clarification_asr.install_archive(archive_path, destination, False)
    assert (destination / "encoder_model.ort").read_bytes() == b"encoder-bytes"
    assert (destination / "decoder_model_merged.ort").read_bytes() == b"decoder-bytes"
    assert (destination / "tokens.txt").read_bytes() == b"token-bytes"
    assert not (destination / "unused.ort").exists()
    assert metadata == json.loads((destination / "installation.json").read_text())
    with pytest.raises(FileExistsError):
        install_clarification_asr.install_archive(archive_path, destination, False)


def test_download_failure_retries_and_reports_a_short_actionable_error(
    tmp_path, monkeypatch
) -> None:
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise urllib.error.URLError("fixture DNS failure")

    monkeypatch.setattr(install_clarification_asr.urllib.request, "urlopen", fail)
    monkeypatch.setattr(install_clarification_asr.time, "sleep", lambda seconds: None)
    with pytest.raises(
        install_clarification_asr.ModelDownloadError,
        match="github.com resolves",
    ):
        install_clarification_asr.download_archive(tmp_path / "model.tar.bz2")
    assert calls == 3
    assert not (tmp_path / "model.tar.bz2").exists()

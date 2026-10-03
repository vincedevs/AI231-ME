from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .config import AlfredConfig


class SpeechRecognitionError(RuntimeError):
    """Expected local speech-to-text failure during a clarification turn."""


def clean_asr_transcript(value: str) -> str:
    """Remove common decoder artifacts without rewriting normal dictation."""
    text = " ".join(value.strip().split())
    # Moonshine can emit either punctuation or its spoken name at the end of a
    # short response. Neither belongs in canonical slot values or messages.
    text = re.sub(r"(?:\s+(?:dot|period|full stop))[.!?]*$", "", text, flags=re.IGNORECASE)
    text = text.rstrip(" .,!?:;")
    # Suppress immediate decoder repetitions such as "for for" while leaving
    # non-adjacent repeated words untouched.
    words = text.split()
    cleaned: list[str] = []
    for word in words:
        comparable = word.strip(".,!?;:").casefold()
        previous = cleaned[-1].strip(".,!?;:").casefold() if cleaned else None
        if comparable and comparable == previous:
            continue
        cleaned.append(word)
    return " ".join(cleaned)


class MoonshineRecognizer:
    """Lazy, offline Moonshine v2 Tiny English recognition via Sherpa-ONNX."""

    def __init__(
        self,
        *,
        encoder_path: Path,
        decoder_path: Path,
        tokens_path: Path,
        sample_rate: int,
        threads: int,
    ) -> None:
        self.encoder_path = encoder_path
        self.decoder_path = decoder_path
        self.tokens_path = tokens_path
        self.sample_rate = int(sample_rate)
        self.threads = int(threads)
        self._recognizer: Any | None = None

    def _load(self) -> Any:
        if self._recognizer is not None:
            return self._recognizer
        missing = [
            path
            for path in (self.encoder_path, self.decoder_path, self.tokens_path)
            if not path.is_file()
        ]
        if missing:
            raise SpeechRecognitionError(
                "The Moonshine clarification model is not installed. "
                "Run tools/install_clarification_asr.py."
            )
        try:
            import sherpa_onnx
        except ImportError as error:
            raise SpeechRecognitionError(
                "Sherpa-ONNX is unavailable for clarification transcription."
            ) from error
        try:
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
                encoder=str(self.encoder_path),
                decoder=str(self.decoder_path),
                tokens=str(self.tokens_path),
                num_threads=self.threads,
                decoding_method="greedy_search",
                provider="cpu",
            )
        except Exception as error:
            raise SpeechRecognitionError(
                "The Moonshine clarification model could not be loaded."
            ) from error
        return self._recognizer

    def transcribe(self, waveform: Any) -> str:
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not bool(np.isfinite(audio).all()):
            raise SpeechRecognitionError("The clarification audio is invalid.")
        recognizer = self._load()
        try:
            stream = recognizer.create_stream()
            stream.accept_waveform(self.sample_rate, audio.clip(-1, 1))
            recognizer.decode_stream(stream)
            return clean_asr_transcript(str(stream.result.text))
        except Exception as error:
            raise SpeechRecognitionError(
                "The clarification response could not be transcribed."
            ) from error


def build_speech_recognizer(config: AlfredConfig) -> MoonshineRecognizer | None:
    values = config.document["clarification"]
    if not bool(values["enabled"]):
        return None
    asr = values["asr"]
    return MoonshineRecognizer(
        encoder_path=config.local_path(str(asr["encoder_path"])),
        decoder_path=config.local_path(str(asr["decoder_path"])),
        tokens_path=config.local_path(str(asr["tokens_path"])),
        sample_rate=int(config.document["audio"]["sample_rate"]),
        threads=int(asr["threads"]),
    )

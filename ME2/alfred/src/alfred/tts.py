from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from .audio import prepare_playback_audio
from .config import AlfredConfig


def prepare_tts_text(value: str) -> str:
    """Remove literal punctuation names and adjacent ASR repetitions."""
    text = " ".join(value.strip().split())
    text = re.sub(r"(?:\s+(?:dot|period|full stop))[.!?]*$", "", text, flags=re.IGNORECASE)
    repeated_word = re.compile(r"\b([A-Za-z]+)(?:\s+\1\b)+", flags=re.IGNORECASE)
    previous = None
    while text != previous:
        previous = text
        text = repeated_word.sub(r"\1", text)
    return text.strip()


class PiperTts:
    """Persistent offline Piper VITS synthesis through sherpa-onnx."""

    def __init__(
        self,
        *,
        model_path: Path,
        tokens_path: Path,
        data_dir: Path,
        threads: int,
        speed: float,
        silence_scale: float,
        playback_gain: float,
        output_device: int | str | None = None,
    ) -> None:
        try:
            import sherpa_onnx
        except ImportError as error:
            raise RuntimeError(f"sherpa-onnx could not be imported: {error}") from error
        for path, label in (
            (model_path, "Piper model"),
            (tokens_path, "Piper tokens"),
            (data_dir, "Piper espeak-ng data"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if threads <= 0:
            raise ValueError("Piper TTS threads must be positive")
        if speed <= 0:
            raise ValueError("Piper TTS speed must be positive")
        if silence_scale < 0:
            raise ValueError("Piper TTS silence_scale cannot be negative")
        if not 0 < playback_gain <= 1:
            raise ValueError("TTS playback gain must be greater than 0 and at most 1")

        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_path),
                    tokens=str(tokens_path),
                    data_dir=str(data_dir),
                ),
                provider="cpu",
                num_threads=threads,
                debug=False,
            ),
            max_num_sentences=1,
        )
        if not tts_config.validate():
            raise ValueError("Piper TTS configuration is invalid")
        self.engine = sherpa_onnx.OfflineTts(tts_config)
        self.generation_config = sherpa_onnx.GenerationConfig()
        self.generation_config.sid = 0
        self.generation_config.speed = speed
        self.generation_config.silence_scale = silence_scale
        self.output_device = output_device
        self.playback_gain = playback_gain

    def generate(self, text: str) -> tuple[np.ndarray, int, dict[str, float]]:
        message = prepare_tts_text(text)
        if not message:
            raise ValueError("TTS text cannot be empty")
        started = time.perf_counter()
        generated = self.engine.generate(message, self.generation_config)
        elapsed = time.perf_counter() - started
        samples = np.asarray(generated.samples, dtype=np.float32)
        sample_rate = int(generated.sample_rate)
        if samples.size == 0 or sample_rate <= 0 or not np.isfinite(samples).all():
            raise RuntimeError("Piper TTS produced invalid audio")
        duration = samples.size / sample_rate
        return (
            samples,
            sample_rate,
            {
                "generation_seconds": elapsed,
                "audio_seconds": duration,
                "real_time_factor": elapsed / duration,
            },
        )

    def speak(self, text: str) -> dict[str, float]:
        try:
            import sounddevice as sd
        except ImportError as error:
            raise RuntimeError("sounddevice is required for TTS playback") from error
        samples, sample_rate, metrics = self.generate(text)
        device_rate = round(
            float(sd.query_devices(self.output_device, kind="output")["default_samplerate"])
        )
        playback = prepare_playback_audio(
            samples,
            sample_rate,
            device_rate,
            self.playback_gain,
        )
        sd.play(playback, device_rate, device=self.output_device, blocking=True)
        return metrics


def build_tts(config: AlfredConfig, output_device: int | str | None = None) -> PiperTts | None:
    values = config.document["tts"]
    if not values["enabled"]:
        return None
    artifact: dict[str, Any] | None = config.model_manifest.get("artifacts", {}).get("tts")
    if not isinstance(artifact, dict):
        raise TypeError("Enabled TTS requires a packaged tts artifact")
    files = artifact.get("files", {})
    directories = artifact.get("directories", {})
    return PiperTts(
        model_path=config.local_path(str(files["model"]["path"])),
        tokens_path=config.local_path(str(files["tokens"]["path"])),
        data_dir=config.local_path(str(directories["espeak_data"]["path"])),
        threads=int(values["threads"]),
        speed=float(values["speed"]),
        silence_scale=float(values["silence_scale"]),
        playback_gain=float(config.document["audio"]["tts_gain"]),
        output_device=output_device,
    )

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .audio import prepare_playback_audio
from .config import AlfredConfig

LOGGER = logging.getLogger(__name__)


def announcement_text(template: str, slots: dict[str, Any]) -> str:
    values = {
        name: str(value.get("surface", "")) if isinstance(value, dict) else str(value)
        for name, value in slots.items()
    }
    try:
        return template.format_map(values)
    except KeyError as error:
        raise ValueError(f"Announcement requires missing slot {error.args[0]!r}") from error


class FeedbackPlayer:
    """Synchronous feedback playback that keeps state transitions explicit."""

    def __init__(
        self,
        config: AlfredConfig,
        output_device: int | str | None = None,
        tts: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.output_device = output_device
        self.playback_gain = float(config.document["audio"]["feedback_gain"])
        self.tts = tts
        self.sleep = sleep
        self.response_segment_pause_seconds = float(
            config.document["tts"]["response_segment_pause_seconds"]
        )
        self.spoken_responses = tts is not None
        self.sounds = dict(config.feedback.get("sounds", {}))
        self.announcements = dict(config.feedback.get("announcements", {}))

    def sound_path(self, event: str) -> Path:
        value = self.sounds.get(event)
        if value is None:
            raise KeyError(f"No feedback sound is configured for {event!r}")
        return self.config.local_path(str(value))

    def play(self, event: str) -> None:
        path = self.sound_path(event)
        if not path.is_file():
            raise FileNotFoundError(f"Missing feedback sound: {path}")
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError("sounddevice and soundfile are required for feedback") from error
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        device_rate = round(
            float(sd.query_devices(self.output_device, kind="output")["default_samplerate"])
        )
        audio = prepare_playback_audio(
            audio,
            sample_rate,
            device_rate,
            self.playback_gain,
        )
        sd.play(audio, device_rate, device=self.output_device, blocking=True)

    def announce(self, intent: str, slots: dict[str, Any]) -> str:
        template = str(self.announcements.get(intent, f"Executing {intent}."))
        message = announcement_text(template, slots)
        LOGGER.info("Alfred: %s", message)
        print(f"Alfred: {message}", flush=True)
        if self.tts is not None:
            metrics = self.tts.speak(message)
            LOGGER.info(
                "tts_complete generation_seconds=%.3f audio_seconds=%.3f rtf=%.3f",
                metrics["generation_seconds"],
                metrics["audio_seconds"],
                metrics["real_time_factor"],
            )
        return message

    def respond(self, message: str) -> str:
        text = message.strip()
        if not text:
            raise ValueError("Feedback response cannot be empty")
        LOGGER.info("Alfred: %s", text)
        print(f"Alfred: {text}", flush=True)
        if self.tts is not None:
            segments = [segment.strip() for segment in text.splitlines() if segment.strip()]
            for index, segment in enumerate(segments):
                self.tts.speak(segment)
                if index + 1 < len(segments):
                    self.sleep(self.response_segment_pause_seconds)
        return text


class NullFeedbackPlayer:
    """Test and headless implementation that records feedback ordering."""

    def __init__(self, announcements: dict[str, str] | None = None) -> None:
        self.events: list[str] = []
        self.announcements = announcements or {}

    def play(self, event: str) -> None:
        self.events.append(f"sound:{event}")

    def announce(self, intent: str, slots: dict[str, Any]) -> str:
        self.events.append(f"announce:{intent}")
        return announcement_text(self.announcements.get(intent, intent), slots)

    def respond(self, message: str) -> str:
        self.events.append(f"response:{message}")
        return message

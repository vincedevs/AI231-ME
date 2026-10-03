from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .audio import int16_to_float32
from .contracts import CaptureResult


class SileroVad:
    """Stateful direct ONNX adapter for the bundled 16 kHz Silero VAD."""

    sample_rate = 16_000

    def __init__(self, model_path: Path, frame_samples: int = 480, threads: int = 1) -> None:
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("onnxruntime is required for Silero VAD") from error
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing Silero VAD model: {model_path}")
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()
        self.frame_samples = int(frame_samples)
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        input_names = {item.name for item in self.session.get_inputs()}
        expected = {"input", "h", "c", "sr"}
        if not expected.issubset(input_names):
            raise ValueError(f"Bundled Silero model has incompatible inputs: {sorted(input_names)}")
        self.reset()

    def reset(self) -> None:
        import numpy as np

        self.hidden = np.zeros((2, 1, 64), dtype=np.float32)
        self.cell = np.zeros((2, 1, 64), dtype=np.float32)

    def score(self, frame: Any) -> float:
        import numpy as np

        audio = np.asarray(frame, dtype=np.float32).reshape(-1)
        if audio.size != self.frame_samples:
            raise ValueError(f"Expected {self.frame_samples} VAD samples, got {audio.size}")
        output, self.hidden, self.cell = self.session.run(
            None,
            {
                "input": audio.clip(-1, 1)[None, :],
                "h": self.hidden,
                "c": self.cell,
                "sr": np.asarray(self.sample_rate, dtype=np.int64),
            },
        )
        return float(output[0, 0])


class CommandCapture:
    """Convert recurrent VAD scores into one bounded command waveform."""

    def __init__(self, vad: Any, config: dict[str, Any]) -> None:
        self.vad = vad
        self.config = config
        frame_ms = vad.frame_samples * 1000 / vad.sample_rate
        self.start_frames = max(1, round(float(config["speech_start_ms"]) / frame_ms))
        self.pre_roll_frames = max(
            self.start_frames, round(float(config["pre_roll_ms"]) / frame_ms)
        )
        self.minimum_speech_frames = max(1, round(float(config["minimum_speech_ms"]) / frame_ms))
        self.end_frames = max(1, round(float(config["end_silence_ms"]) / frame_ms))
        self.post_roll_frames = max(0, round(float(config["post_roll_ms"]) / frame_ms))
        self.timeout_frames = max(
            1,
            round(
                float(config["speech_start_timeout_seconds"]) * vad.sample_rate / vad.frame_samples
            ),
        )
        self.maximum_frames = max(
            1,
            round(float(config["maximum_command_seconds"]) * vad.sample_rate / vad.frame_samples),
        )

    def capture(
        self,
        read_frame: Callable[[], Any],
        on_speech_start: Callable[[], None] | None = None,
    ) -> CaptureResult:
        import numpy as np

        self.vad.reset()
        pre_roll: deque[Any] = deque(maxlen=self.pre_roll_frames)
        captured: list[Any] = []
        trigger_frames = 0
        speech_frames = 0
        silence_frames = 0
        active = False

        for frame_index in range(self.timeout_frames + self.maximum_frames):
            frame = np.asarray(read_frame()).reshape(-1)
            if frame.size != self.vad.frame_samples:
                raise ValueError(
                    f"Command capture expected {self.vad.frame_samples} samples, got {frame.size}"
                )
            audio = (
                int16_to_float32(frame)
                if frame.dtype.kind in {"i", "u"}
                else frame.astype(np.float32)
            )
            score = self.vad.score(audio)
            speech = score >= float(self.config["vad_threshold"])

            if not active:
                pre_roll.append(audio.copy())
                trigger_frames = trigger_frames + 1 if speech else 0
                if trigger_frames >= self.start_frames:
                    active = True
                    captured = list(pre_roll)
                    speech_frames = trigger_frames
                    if on_speech_start is not None:
                        on_speech_start()
                elif frame_index + 1 >= self.timeout_frames:
                    return CaptureResult(None, "no_speech")
                continue

            captured.append(audio.copy())
            if speech:
                speech_frames += 1
                silence_frames = 0
            else:
                silence_frames += 1

            reached_end = silence_frames >= self.end_frames
            reached_limit = len(captured) >= self.maximum_frames
            if reached_end or reached_limit:
                if speech_frames < self.minimum_speech_frames:
                    return CaptureResult(None, "speech_too_short")
                if reached_end and silence_frames > self.post_roll_frames:
                    captured = captured[: -(silence_frames - self.post_roll_frames)]
                waveform = np.concatenate(captured).astype(np.float32, copy=False)
                return CaptureResult(
                    waveform,
                    "trailing_silence" if reached_end else "maximum_duration",
                    waveform.size / self.vad.sample_rate,
                )

        return CaptureResult(None, "capture_exhausted")

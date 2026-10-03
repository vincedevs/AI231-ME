# Portions of the streaming feature extraction are adapted from openWakeWord.
# Copyright 2022 David Scripka. Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DetectionGate:
    threshold: float
    required_hits: int
    cooldown_seconds: float
    consecutive_hits: int = 0
    last_detection_time: float = float("-inf")

    def update(self, score: float, now: float) -> bool:
        self.consecutive_hits = self.consecutive_hits + 1 if score >= self.threshold else 0
        if self.consecutive_hits < self.required_hits:
            return False
        self.consecutive_hits = 0
        if now - self.last_detection_time < self.cooldown_seconds:
            return False
        self.last_detection_time = now
        return True


def _cpu_session(model_path: Path, threads: int = 1) -> Any:
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime is required for wake-word detection") from error

    if hasattr(ort, "disable_telemetry_events"):
        ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.inter_op_num_threads = threads
    options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class WakeWordFeatures:
    """Streaming openWakeWord-compatible ONNX frontend for 16 kHz int16 PCM."""

    chunk_samples = 1280
    mel_window_frames = 76
    mel_step_frames = 8
    embedding_dimensions = 96

    def __init__(self, melspectrogram_model: Path, embedding_model: Path) -> None:
        self.melspectrogram_session = _cpu_session(melspectrogram_model)
        self.embedding_session = _cpu_session(embedding_model)
        self.melspectrogram_input = self.melspectrogram_session.get_inputs()[0].name
        self.embedding_input = self.embedding_session.get_inputs()[0].name
        self.raw_buffer: deque[int] = deque(maxlen=16000 * 10)
        self.melspectrogram_buffer: Any
        self.feature_buffer: Any
        self._initial_features: Any
        self._initialize_buffers()

    def _melspectrogram(self, samples: Any) -> Any:
        import numpy as np

        values = np.asarray(samples)
        if values.dtype != np.int16:
            raise ValueError("Wake-word audio must be 16-bit PCM")
        values = values.reshape(1, -1).astype(np.float32)
        output = self.melspectrogram_session.run(None, {self.melspectrogram_input: values})[0]
        return np.squeeze(output) / 10.0 + 2.0

    def _embeddings(self, samples: Any) -> Any:
        import numpy as np

        spectrogram = self._melspectrogram(samples)
        windows = [
            spectrogram[start : start + self.mel_window_frames]
            for start in range(0, spectrogram.shape[0], self.mel_step_frames)
            if spectrogram[start : start + self.mel_window_frames].shape[0]
            == self.mel_window_frames
        ]
        if not windows:
            return np.empty((0, self.embedding_dimensions), dtype=np.float32)
        model_input = np.expand_dims(np.asarray(windows), axis=-1).astype(np.float32)
        return np.asarray(
            self.embedding_session.run(None, {self.embedding_input: model_input})[0]
        ).reshape(-1, self.embedding_dimensions)

    def _initialize_buffers(self) -> None:
        import numpy as np

        self.raw_buffer.clear()
        self.melspectrogram_buffer = np.ones((self.mel_window_frames, 32), dtype=np.float32)
        # Match openWakeWord's warm-up context while making startup reproducible.
        random_audio = np.random.default_rng(0).integers(-1000, 1000, 16000 * 4, dtype=np.int16)
        self._initial_features = self._embeddings(random_audio)
        self.feature_buffer = self._initial_features.copy()

    def reset(self) -> None:
        import numpy as np

        self.raw_buffer.clear()
        self.melspectrogram_buffer = np.ones((self.mel_window_frames, 32), dtype=np.float32)
        self.feature_buffer = self._initial_features.copy()

    def update(self, samples: Any) -> None:
        import numpy as np

        values = np.asarray(samples)
        if values.dtype != np.int16 or values.ndim != 1:
            raise ValueError("Wake-word chunks must be one-dimensional int16 PCM")
        if values.size != self.chunk_samples:
            raise ValueError(
                f"Wake-word chunks must contain {self.chunk_samples} samples; got {values.size}"
            )

        self.raw_buffer.extend(values.tolist())
        # The extra 480 samples reproduce the overlap used by openWakeWord's
        # streaming mel frontend. Each 80 ms chunk yields eight new mel frames.
        recent = np.asarray(list(self.raw_buffer)[-self.chunk_samples - 480 :], dtype=np.int16)
        new_mels = np.asarray(self._melspectrogram(recent), dtype=np.float32).reshape(-1, 32)
        self.melspectrogram_buffer = np.vstack((self.melspectrogram_buffer, new_mels))[-970:]
        embedding_input = self.melspectrogram_buffer[-self.mel_window_frames :][
            None, :, :, None
        ].astype(np.float32)
        embedding = self.embedding_session.run(None, {self.embedding_input: embedding_input})[0]
        self.feature_buffer = np.vstack(
            (
                self.feature_buffer,
                np.asarray(embedding).reshape(-1, self.embedding_dimensions),
            )
        )[-120:]

    def latest(self, frame_count: int) -> Any:
        import numpy as np

        if frame_count <= 0:
            raise ValueError("Wake-word classifier frame count must be positive")
        if self.feature_buffer.shape[0] < frame_count:
            raise RuntimeError("Insufficient wake-word feature context")
        return np.asarray(self.feature_buffer[-frame_count:], dtype=np.float32)[None, :, :]


class WakeWordClassifier:
    """One binary ONNX classifier with startup prediction suppression."""

    def __init__(self, model_path: Path) -> None:
        self.session = _cpu_session(model_path)
        model_input = self.session.get_inputs()[0]
        if len(model_input.shape) != 3 or not isinstance(model_input.shape[1], int):
            raise ValueError(f"Unsupported wake-word classifier input shape: {model_input.shape}")
        self.input_name = model_input.name
        self.feature_frames = int(model_input.shape[1])
        self.prediction_count = 0

    def predict(self, features: Any) -> float:
        import numpy as np

        output = self.session.run(None, {self.input_name: features})[0]
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        if values.size != 1:
            raise ValueError(f"Expected one wake-word score, received shape {output.shape}")
        score = float(values[0])
        self.prediction_count += 1
        # Match the audited upstream behavior by suppressing startup scores.
        return 0.0 if self.prediction_count <= 5 else score

    def reset(self) -> None:
        self.prediction_count = 0


class DualWakeWordDetector:
    """Run two classifiers over one shared ONNX feature frontend."""

    def __init__(
        self,
        *,
        hey_alfred_model: Path,
        im_batman_model: Path,
        melspectrogram_model: Path,
        embedding_model: Path,
        hey_alfred_threshold: float,
        im_batman_threshold: float,
        required_hits: int,
        cooldown_seconds: float,
    ) -> None:
        for path in (
            hey_alfred_model,
            im_batman_model,
            melspectrogram_model,
            embedding_model,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Missing wake-word artifact: {path}")
        self.features = WakeWordFeatures(melspectrogram_model, embedding_model)
        self.classifiers = {
            "hey_alfred": WakeWordClassifier(hey_alfred_model),
            "im_batman": WakeWordClassifier(im_batman_model),
        }
        self.gates = {
            "hey_alfred": DetectionGate(hey_alfred_threshold, required_hits, cooldown_seconds),
            "im_batman": DetectionGate(im_batman_threshold, required_hits, cooldown_seconds),
        }
        self.last_scores = {name: 0.0 for name in self.classifiers}

    def process(self, samples: Any, now: float | None = None) -> str | None:
        self.features.update(samples)
        timestamp = time.monotonic() if now is None else now
        detected: list[tuple[str, float]] = []
        for name, classifier in self.classifiers.items():
            score = classifier.predict(self.features.latest(classifier.feature_frames))
            self.last_scores[name] = score
            if self.gates[name].update(score, timestamp):
                detected.append((name, score))
        if not detected:
            return None
        return max(detected, key=lambda item: item[1])[0]

    def reset(self) -> None:
        self.features.reset()
        for classifier in self.classifiers.values():
            classifier.reset()
        for gate in self.gates.values():
            gate.consecutive_hits = 0

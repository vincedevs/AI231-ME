from __future__ import annotations

import numpy as np

from alfred.vad import CommandCapture


class FakeVad:
    sample_rate = 16_000
    frame_samples = 480

    def __init__(self, scores: list[float]) -> None:
        self.scores = iter(scores)
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def score(self, frame: np.ndarray) -> float:
        assert frame.shape == (480,)
        return next(self.scores)


def capture_config() -> dict[str, float]:
    return {
        "vad_threshold": 0.5,
        "speech_start_ms": 60,
        "pre_roll_ms": 90,
        "minimum_speech_ms": 60,
        "end_silence_ms": 90,
        "post_roll_ms": 30,
        "speech_start_timeout_seconds": 0.3,
        "maximum_command_seconds": 1.0,
    }


def test_command_capture_keeps_preroll_and_trims_silence() -> None:
    scores = [0.0, 0.0, 0.8, 0.9, 0.9, 0.0, 0.0, 0.0]
    detector = CommandCapture(FakeVad(scores), capture_config())
    frame = np.ones(480, dtype=np.int16)
    events: list[str] = []
    result = detector.capture(lambda: frame, lambda: events.append("speech"))
    assert result.waveform is not None
    assert result.reason == "trailing_silence"
    assert events == ["speech"]
    # Three pre-roll frames (including both trigger frames), one following
    # speech frame, and one configured post-roll silence frame are retained.
    assert result.waveform.size == 5 * 480


def test_command_capture_times_out_without_speech() -> None:
    detector = CommandCapture(FakeVad([0.0] * 20), capture_config())
    result = detector.capture(lambda: np.zeros(480, dtype=np.int16))
    assert result.waveform is None
    assert result.reason == "no_speech"

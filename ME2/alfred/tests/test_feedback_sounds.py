from __future__ import annotations

import hashlib
import struct
import wave

from alfred.config import load_config

XYLOPHONE_EVENTS = (
    "action_failure",
    "action_success",
    "listening_end",
    "listening_start",
    "low_confidence",
    "unsupported",
)


def test_xylophone_feedback_assets_are_distinct_smooth_pcm() -> None:
    config = load_config()
    checksums = set()
    for event in XYLOPHONE_EVENTS:
        path = config.local_path(config.feedback["sounds"][event])
        checksums.add(hashlib.sha256(path.read_bytes()).hexdigest())
        with wave.open(str(path), "rb") as stream:
            assert stream.getnchannels() == 1
            assert stream.getsampwidth() == 2
            assert stream.getframerate() == 44_100
            assert 0.5 <= stream.getnframes() / stream.getframerate() <= 0.9
            frames = stream.readframes(stream.getnframes())
        samples = struct.unpack(f"<{len(frames) // 2}h", frames)
        peak = max(abs(value) for value in samples) / 32767
        assert 0.5 <= peak <= 0.8
        assert abs(samples[0]) <= 1
        assert abs(samples[-1]) <= 1
    assert len(checksums) == len(XYLOPHONE_EVENTS)

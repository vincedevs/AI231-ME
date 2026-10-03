from __future__ import annotations

import numpy as np
import pytest

from alfred.audio import (
    AudioFramer,
    int16_to_float32,
    matching_device_indices,
    playback_output_device,
    prepare_playback_audio,
    resample_for_output,
)


def test_audio_framer_preserves_every_sample() -> None:
    framer = AudioFramer(4)
    assert framer.push(np.asarray([0, 1, 2], dtype=np.int16)) == []
    frames = framer.push(np.asarray([3, 4, 5, 6, 7, 8], dtype=np.int16))
    assert [frame.tolist() for frame in frames] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    frames = framer.push(np.asarray([9, 10, 11], dtype=np.int16))
    assert [frame.tolist() for frame in frames] == [[8, 9, 10, 11]]


def test_int16_conversion_uses_pcm_scale() -> None:
    converted = int16_to_float32(np.asarray([-32768, 0, 32767], dtype=np.int16))
    np.testing.assert_allclose(converted, [-1.0, 0.0, 32767 / 32768])


def test_matching_audio_device_prefers_one_full_duplex_respeaker() -> None:
    devices = [
        {"name": "Built-in Audio", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "ReSpeaker Lite USB", "max_input_channels": 2, "max_output_channels": 2},
    ]
    assert matching_device_indices(devices, "respeaker") == (1, 1)


def test_matching_audio_device_rejects_an_incomplete_match() -> None:
    devices = [{"name": "ReSpeaker microphone", "max_input_channels": 2, "max_output_channels": 0}]
    with pytest.raises(RuntimeError, match="full-duplex"):
        matching_device_indices(devices, "ReSpeaker")


def test_output_resampling_preserves_channels_and_duration() -> None:
    stereo = np.column_stack((np.linspace(-1, 1, 100, dtype=np.float32), np.linspace(1, -1, 100)))
    converted = resample_for_output(stereo, 20_000, 16_000)
    assert converted.shape == (80, 2)
    assert converted.dtype == np.float32
    np.testing.assert_allclose(converted[[0, -1]], stereo[[0, -1]])


def test_common_playback_processing_applies_gain_fades_and_limiter() -> None:
    audio = np.ones(100, dtype=np.float32)
    prepared = prepare_playback_audio(audio, 1000, 1000, 0.15, fade_ms=10, trailing_silence_ms=20)
    assert prepared.shape == (120,)
    assert prepared[0] == 0
    assert prepared[-1] == 0
    assert np.max(np.abs(prepared)) <= 0.15


def test_system_default_output_preserves_operating_system_volume_control() -> None:
    assert playback_output_device({"use_system_default_output": True, "output_device": 7}) is None
    assert playback_output_device({"use_system_default_output": False, "output_device": 7}) == 7

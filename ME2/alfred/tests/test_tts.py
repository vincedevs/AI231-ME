from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from alfred.tts import PiperTts, prepare_tts_text


class FakeTtsConfig:
    def __init__(self, **values) -> None:
        self.values = values

    def validate(self) -> bool:
        return True


class FakeEngine:
    def __init__(self, config) -> None:
        self.config = config

    def generate(self, text, generation_config):
        assert text == "Hello."
        assert generation_config.speed == 1.0
        return SimpleNamespace(samples=[0.0, 0.25, -0.25], sample_rate=22_050)


class FakeGenerationConfig:
    pass


class FakeSoundDevice:
    def __init__(self) -> None:
        self.played = None

    def play(self, samples, sample_rate, *, device, blocking) -> None:
        self.played = (samples, sample_rate, device, blocking)

    def query_devices(self, device, *, kind):
        assert device is None
        assert kind == "output"
        return {"default_samplerate": 16_000}


def test_piper_generates_finite_audio_without_playback(monkeypatch, tmp_path) -> None:
    model = tmp_path / "model.onnx"
    tokens = tmp_path / "tokens.txt"
    data = tmp_path / "espeak-ng-data"
    model.touch()
    tokens.touch()
    data.mkdir()
    fake_module = SimpleNamespace(
        OfflineTtsConfig=FakeTtsConfig,
        OfflineTtsModelConfig=FakeTtsConfig,
        OfflineTtsVitsModelConfig=FakeTtsConfig,
        OfflineTts=FakeEngine,
        GenerationConfig=FakeGenerationConfig,
    )
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_module)
    tts = PiperTts(
        model_path=model,
        tokens_path=tokens,
        data_dir=data,
        threads=2,
        speed=1.0,
        silence_scale=0.2,
        playback_gain=0.5,
    )
    samples, sample_rate, metrics = tts.generate("Hello.")
    assert samples.dtype == np.float32
    assert sample_rate == 22_050
    assert metrics["audio_seconds"] > 0
    assert metrics["real_time_factor"] >= 0

    sounddevice = FakeSoundDevice()
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    tts.speak("Hello.")
    playback, playback_rate, device, blocking = sounddevice.played
    assert playback_rate == 16_000
    assert device is None
    assert blocking is True
    assert playback.size > samples.size
    assert np.max(np.abs(playback)) <= 0.125
    assert playback[0] == 0
    assert playback[-1] == 0


def test_tts_text_removes_decoder_punctuation_and_immediate_repetitions() -> None:
    assert prepare_tts_text("Setting a timer for for four minutes dot.") == (
        "Setting a timer for four minutes"
    )
    assert prepare_tts_text("Please call Dad. Please confirm.") == (
        "Please call Dad. Please confirm."
    )

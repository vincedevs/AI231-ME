#!/usr/bin/env python3
"""Generate Alfred's deterministic tone and xylophone feedback sounds."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "assets" / "sounds"
SAMPLE_RATE = 16_000
XYLOPHONE_SAMPLE_RATE = 44_100


SOUNDS = {
    "easter_egg_1.wav": (261.63, 0.28, 0.30),
    "easter_egg_2.wav": (329.63, 0.28, 0.30),
    "easter_egg_3.wav": (392.00, 0.28, 0.30),
    "easter_egg_4.wav": (523.25, 0.28, 0.30),
    "easter_egg_5.wav": (783.99, 0.28, 0.30),
}

# MIDI notes keep the musical relationships visible and avoid hidden frequency
# constants. C5=72, E-flat5=75, E5=76, F5=77, G5=79, A5=81, and B4=71.
# Each strike is (MIDI note, onset seconds, velocity). Durations include the
# final wooden-bar decay and a short room tail.
XYLOPHONE_CUES = {
    "action_success.wav": {
        "strikes": [(72, 0.00, 0.84), (76, 0.10, 0.88), (79, 0.20, 0.94)],
        "duration": 0.68,
        "brightness": 1.05,
        "peak": 0.78,
    },
    "action_failure.wav": {
        "strikes": [(79, 0.00, 0.78), (75, 0.13, 0.72), (72, 0.27, 0.76)],
        "duration": 0.76,
        "brightness": 0.82,
        "peak": 0.70,
    },
    "listening_start.wav": {
        "strikes": [(72, 0.00, 0.72), (79, 0.12, 0.84)],
        "duration": 0.55,
        "brightness": 1.00,
        "peak": 0.68,
    },
    "listening_end.wav": {
        "strikes": [(79, 0.00, 0.74), (72, 0.13, 0.80)],
        "duration": 0.59,
        "brightness": 0.88,
        "peak": 0.66,
    },
    "low_confidence.wav": {
        "strikes": [(76, 0.00, 0.55), (77, 0.16, 0.50), (76, 0.33, 0.52)],
        "duration": 0.78,
        "brightness": 0.72,
        "peak": 0.55,
    },
    "unsupported.wav": {
        "strikes": [(71, 0.00, 0.62), (77, 0.17, 0.68), (71, 0.35, 0.60)],
        "duration": 0.82,
        "brightness": 0.90,
        "peak": 0.62,
    },
}


def write_tone(path: Path, frequency: float, duration: float, amplitude: float) -> None:
    samples = round(SAMPLE_RATE * duration)
    fade = max(1, round(SAMPLE_RATE * min(0.02, duration / 4)))
    frames = bytearray()
    for index in range(samples):
        envelope = min(1.0, index / fade, (samples - index - 1) / fade)
        value = (
            amplitude * max(0.0, envelope) * math.sin(2 * math.pi * frequency * index / SAMPLE_RATE)
        )
        frames.extend(struct.pack("<h", round(value * 32767)))
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(frames)


def midi_frequency(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def xylophone_strike(
    frequency: float,
    duration: float,
    velocity: float,
    brightness: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Synthesize one struck wooden bar using its principal vibration modes."""
    samples = max(1, round(duration * XYLOPHONE_SAMPLE_RATE))
    time_axis = np.arange(samples, dtype=np.float64) / XYLOPHONE_SAMPLE_RATE

    # Ratios approximate the first modes of a freely vibrating bar. Higher
    # modes are quieter and damp much faster, producing the characteristic
    # hard-mallet attack without a sustained electronic-sounding tone.
    modes = (
        (1.000, 1.00, 0.36),
        (2.756, 0.34 * brightness, 0.18),
        (5.404, 0.15 * brightness, 0.095),
        (8.933, 0.065 * brightness, 0.055),
    )
    signal = np.zeros(samples, dtype=np.float64)
    strike_detune = 0.0035
    settling_seconds = 0.018
    for mode_index, (ratio, amplitude, decay_seconds) in enumerate(modes):
        mode_frequency = frequency * ratio
        # Integrating a quickly decaying pitch offset gives a natural, subtle
        # settling motion immediately after the mallet hits the bar.
        phase_cycles = mode_frequency * (
            time_axis
            + strike_detune * settling_seconds * (1.0 - np.exp(-time_axis / settling_seconds))
        )
        phase = 2.0 * math.pi * phase_cycles + mode_index * 0.21
        envelope = (1.0 - np.exp(-time_axis / 0.0012)) * np.exp(-time_axis / decay_seconds)
        signal += amplitude * envelope * np.sin(phase)

    # A very short, deterministic broadband component represents the mallet
    # contact. Its decay is fast enough to avoid hiss or a click at playback.
    contact_samples = min(samples, round(0.013 * XYLOPHONE_SAMPLE_RATE))
    contact_time = np.arange(contact_samples, dtype=np.float64) / XYLOPHONE_SAMPLE_RATE
    contact = generator.normal(0.0, 1.0, contact_samples)
    contact *= (1.0 - np.exp(-contact_time / 0.00045)) * np.exp(-contact_time / 0.0022)
    signal[:contact_samples] += 0.075 * brightness * contact
    return velocity * signal


def synthesize_xylophone_cue(name: str, specification: dict) -> np.ndarray:
    samples = round(float(specification["duration"]) * XYLOPHONE_SAMPLE_RATE)
    output = np.zeros(samples, dtype=np.float64)
    seed = int.from_bytes(name.encode("utf-8"), "little") % 2**32
    generator = np.random.default_rng(seed)
    brightness = float(specification["brightness"])
    for note, onset, velocity in specification["strikes"]:
        start = round(float(onset) * XYLOPHONE_SAMPLE_RATE)
        strike = xylophone_strike(
            midi_frequency(int(note)),
            (samples - start) / XYLOPHONE_SAMPLE_RATE,
            float(velocity),
            brightness,
            generator,
        )
        output[start : start + strike.size] += strike

    # Quiet early reflections add depth without turning a short UI cue into a
    # reverberant sound. Reusing the dry signal keeps generation deterministic.
    dry = output.copy()
    for delay_seconds, gain in ((0.013, 0.105), (0.029, 0.060), (0.047, 0.032)):
        delay = round(delay_seconds * XYLOPHONE_SAMPLE_RATE)
        output[delay:] += gain * dry[:-delay]

    output -= np.mean(output)
    fade_in = min(samples, round(0.0015 * XYLOPHONE_SAMPLE_RATE))
    fade_out = min(samples, round(0.030 * XYLOPHONE_SAMPLE_RATE))
    output[:fade_in] *= np.sin(np.linspace(0.0, math.pi / 2.0, fade_in)) ** 2
    output[-fade_out:] *= np.cos(np.linspace(0.0, math.pi / 2.0, fade_out)) ** 2
    peak = float(np.max(np.abs(output)))
    if not math.isfinite(peak) or peak <= 0:
        raise RuntimeError(f"Synthesized invalid audio for {name}")
    output *= float(specification["peak"]) / peak
    return output.astype(np.float32)


def write_pcm16(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError(f"Cannot write invalid mono audio to {path}")
    pcm = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, parameters in SOUNDS.items():
        write_tone(OUTPUT / name, *parameters)
        print(f"Wrote {OUTPUT / name}")
    for name, specification in XYLOPHONE_CUES.items():
        write_pcm16(
            OUTPUT / name,
            synthesize_xylophone_cue(name, specification),
            XYLOPHONE_SAMPLE_RATE,
        )
        print(f"Wrote {OUTPUT / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

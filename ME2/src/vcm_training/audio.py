from __future__ import annotations

import math
import random
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _hertz_to_mel(frequency: float) -> float:
    return 2595.0 * math.log10(1.0 + frequency / 700.0)


def _mel_to_hertz(value: float) -> float:
    return 700.0 * (10.0 ** (value / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    minimum_hertz: float,
    maximum_hertz: float,
) -> torch.Tensor:
    """Return an HTK-style triangular Mel filterbank [mels, fft_bins]."""
    minimum_mel = _hertz_to_mel(minimum_hertz)
    maximum_mel = _hertz_to_mel(maximum_hertz)
    mel_points = torch.linspace(minimum_mel, maximum_mel, n_mels + 2)
    frequencies = torch.tensor([_mel_to_hertz(float(value)) for value in mel_points])
    fft_frequencies = torch.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    filters = torch.zeros(n_mels, n_fft // 2 + 1)
    for index in range(n_mels):
        left, center, right = frequencies[index : index + 3]
        rising = (fft_frequencies - left) / max(float(center - left), 1e-12)
        falling = (right - fft_frequencies) / max(float(right - center), 1e-12)
        filters[index] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return filters


class LogMelFrontend(nn.Module):
    """ONNX-friendly fixed log-Mel frontend operating on raw waveforms.

    A fixed Conv1d implements the real and imaginary DFT bases. This avoids a
    separate application-side feature extractor and avoids complex-valued FFT
    tensors, which are less portable across ONNX runtimes.
    """

    def __init__(self, audio_config: dict[str, Any]) -> None:
        super().__init__()
        self.sample_rate = int(audio_config["sample_rate"])
        self.n_fft = int(audio_config["n_fft"])
        self.hop_length = int(audio_config["hop_length"])
        self.n_mels = int(audio_config["n_mels"])

        window = torch.hann_window(self.n_fft, periodic=True)
        sample_positions = torch.arange(self.n_fft, dtype=torch.float32)
        frequencies = torch.arange(self.n_fft // 2 + 1, dtype=torch.float32).unsqueeze(1)
        phase = 2.0 * math.pi * frequencies * sample_positions / self.n_fft
        real = torch.cos(phase) * window
        imaginary = -torch.sin(phase) * window
        kernels = torch.cat((real, imaginary), dim=0).unsqueeze(1)
        self.register_buffer("dft_kernels", kernels, persistent=True)
        self.register_buffer(
            "mel_filters",
            mel_filterbank(
                self.sample_rate,
                self.n_fft,
                self.n_mels,
                float(audio_config["minimum_hertz"]),
                float(audio_config["maximum_hertz"]),
            ),
            persistent=True,
        )

    def forward(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        lengths = lengths.to(dtype=torch.long).clamp(min=1, max=waveform.shape[1])
        sample_index = torch.arange(waveform.shape[1], device=waveform.device)
        sample_mask = sample_index.unsqueeze(0) < lengths.unsqueeze(1)
        masked_waveform = waveform * sample_mask.to(waveform.dtype)
        peak = masked_waveform.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
        waveform = masked_waveform / peak
        spectrum = F.conv1d(
            waveform.unsqueeze(1),
            self.dft_kernels,
            stride=self.hop_length,
        )
        bins = self.n_fft // 2 + 1
        power = spectrum[:, :bins].square() + spectrum[:, bins:].square()
        mel = torch.einsum("mf,bft->bmt", self.mel_filters, power)
        features = torch.log(mel.clamp(min=1e-6))

        frame_lengths = (
            torch.div(
                (lengths - self.n_fft).clamp(min=0),
                self.hop_length,
                rounding_mode="floor",
            )
            + 1
        )
        frame_lengths = frame_lengths.clamp(max=features.shape[-1])
        frame_index = torch.arange(features.shape[-1], device=features.device)
        frame_mask = frame_index.unsqueeze(0) < frame_lengths.unsqueeze(1)
        mask = frame_mask.unsqueeze(1).to(features.dtype)
        denominator = frame_lengths.to(features.dtype).view(-1, 1, 1).clamp(min=1.0)
        mean = (features * mask).sum(dim=2, keepdim=True) / denominator
        variance = ((features - mean).square() * mask).sum(dim=2, keepdim=True) / denominator
        features = (features - mean) / torch.sqrt(variance + 1e-5)
        features = features * mask
        return features.unsqueeze(1), frame_lengths


def augment_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    config: dict[str, Any],
    rng: random.Random,
) -> torch.Tensor:
    """Apply reproducible training-only gain, shift, and Gaussian-noise augmentation."""
    if rng.random() >= float(config["probability"]):
        return waveform
    minimum_gain, maximum_gain = map(float, config["gain_db"])
    gain = 10.0 ** (rng.uniform(minimum_gain, maximum_gain) / 20.0)
    waveform = waveform * gain
    maximum_shift = round(float(config["time_shift_ms"]) * sample_rate / 1000.0)
    if maximum_shift > 0 and waveform.numel() > 1:
        shift = rng.randint(-maximum_shift, maximum_shift)
        waveform = torch.roll(waveform, shifts=shift)
        if shift > 0:
            waveform[:shift] = 0
        elif shift < 0:
            waveform[shift:] = 0
    minimum_snr, maximum_snr = map(float, config["gaussian_noise_snr_db"])
    signal_rms = waveform.square().mean().sqrt()
    if float(signal_rms) > 1e-8:
        snr = rng.uniform(minimum_snr, maximum_snr)
        noise_rms = signal_rms / (10.0 ** (snr / 20.0))
        generator = torch.Generator().manual_seed(rng.randrange(2**31))
        noise = torch.randn(waveform.shape, generator=generator, dtype=waveform.dtype)
        waveform = waveform + noise * noise_rms
    return waveform.clamp(-1.0, 1.0)

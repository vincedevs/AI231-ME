from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .audio import LogMelFrontend

ARCHITECTURES = (
    "dscnn",
    "tcresnet",
    "bcresnet",
    "crnn_attention",
    "tiny_conformer",
)
SLOT_CHARACTERS = " abcdefghijklmnopqrstuvwxyz0123456789%'"
SLOT_TOKENS = ("<blank>", *SLOT_CHARACTERS)
SLOT_TO_INDEX = {character: index for index, character in enumerate(SLOT_TOKENS)}


def convolution_length(
    lengths: torch.Tensor, kernel: int, stride: int, padding: int, dilation: int = 1
) -> torch.Tensor:
    return (
        torch.div(
            lengths + 2 * padding - dilation * (kernel - 1) - 1,
            stride,
            rounding_mode="floor",
        )
        + 1
    )


def sequence_mask(lengths: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.arange(steps, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class DepthwiseSeparable2d(nn.Module):
    def __init__(self, channels: int, stride: tuple[int, int] = (1, 1)) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, stride=stride, padding=1, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.normalization = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(self.normalization(self.pointwise(self.depthwise(inputs))))


class DSCNNEncoder(nn.Module):
    output_size = 64

    def __init__(self, n_mels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparable2d(64),
            DepthwiseSeparable2d(64, stride=(1, 2)),
            DepthwiseSeparable2d(64),
            DepthwiseSeparable2d(64),
        )

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.blocks(self.stem(features))
        lengths = convolution_length(lengths, 3, 2, 1)
        lengths = convolution_length(lengths, 3, 2, 1)
        return output.mean(dim=2).transpose(1, 2), lengths


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, stride: int = 1, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.first = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
        )
        self.second = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1
            else nn.Sequential(
                nn.Conv1d(channels, channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(channels),
            )
        )
        self.stride = stride
        self.dilation = dilation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(self.second(self.first(inputs)) + self.skip(inputs))


class TCResNetEncoder(nn.Module):
    output_size = 64

    def __init__(self, n_mels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            TemporalResidualBlock(64, dilation=1),
            TemporalResidualBlock(64, stride=2, dilation=2),
            TemporalResidualBlock(64, dilation=4),
            TemporalResidualBlock(64, stride=2, dilation=8),
            TemporalResidualBlock(64, dilation=16),
        )

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.blocks(self.stem(features.squeeze(1)))
        lengths = convolution_length(lengths, 3, 2, 2, dilation=2)
        lengths = convolution_length(lengths, 3, 2, 8, dilation=8)
        return output.transpose(1, 2), lengths


class BroadcastResidualBlock(nn.Module):
    """Frequency processing plus broadcast temporal context, following BC-ResNet."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.frequency = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        frequency = self.frequency(inputs)
        temporal = self.temporal(frequency.mean(dim=2)).unsqueeze(2)
        return F.silu(inputs + frequency + temporal)


class BCResNetEncoder(nn.Module):
    output_size = 64

    def __init__(self, n_mels: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.first = nn.Sequential(
            BroadcastResidualBlock(64, 1),
            BroadcastResidualBlock(64, 2),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=(1, 2), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.second = nn.Sequential(
            BroadcastResidualBlock(64, 4),
            BroadcastResidualBlock(64, 8),
        )

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.second(self.downsample(self.first(self.stem(features))))
        lengths = convolution_length(lengths, 3, 2, 1)
        lengths = convolution_length(lengths, 3, 2, 1)
        return output.mean(dim=2).transpose(1, 2), lengths


class CRNNAttentionEncoder(nn.Module):
    output_size = 192

    def __init__(self, n_mels: int) -> None:
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(),
        )
        reduced_mels = math.ceil(math.ceil(n_mels / 2) / 2)
        self.recurrent = nn.GRU(
            48 * reduced_mels,
            96,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.convolutions(features)
        lengths = convolution_length(lengths, 3, 2, 1)
        lengths = convolution_length(lengths, 3, 2, 1)
        output = output.permute(0, 3, 1, 2).flatten(2)
        output, _ = self.recurrent(output)
        return output, lengths


class ConformerConvolution(nn.Module):
    def __init__(self, dimension: int, kernel_size: int = 15) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(dimension)
        self.pointwise_in = nn.Conv1d(dimension, 2 * dimension, 1)
        self.depthwise = nn.Conv1d(
            dimension,
            dimension,
            kernel_size,
            padding=kernel_size // 2,
            groups=dimension,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm1d(dimension)
        self.pointwise_out = nn.Conv1d(dimension, dimension, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer_norm(inputs).transpose(1, 2)
        output = F.glu(self.pointwise_in(output), dim=1)
        output = F.silu(self.batch_norm(self.depthwise(output)))
        return self.pointwise_out(output).transpose(1, 2)


class ExportableSelfAttention(nn.Module):
    """Multi-head self-attention with dynamic ONNX-safe reshape operations.

    Parameter names and tensor layout match ``nn.MultiheadAttention`` so
    checkpoints trained before this deployment fix remain loadable.
    """

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("Attention dimension must be divisible by its head count")
        self.dimension = dimension
        self.heads = heads
        self.head_dimension = dimension // heads
        self.dropout = dropout
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dimension, dimension))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * dimension))
        self.out_proj = nn.Linear(dimension, dimension)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        projected = F.linear(inputs, self.in_proj_weight, self.in_proj_bias)
        query, key, value = projected.chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            # Preserve the dynamic batch and time axes instead of passing
            # Python shape values to reshape. The legacy form was traced with
            # the 2-second example's 50 frames and failed when ONNX Runtime
            # received the 25 or 75 frames produced by 1- or 3-second audio.
            return tensor.unflatten(-1, (self.heads, self.head_dimension)).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        scores = scores.masked_fill(padding_mask[:, None, None, :], -1e4)
        weights = F.dropout(torch.softmax(scores, dim=-1), self.dropout, self.training)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).flatten(2, 3)
        return self.out_proj(attended)


class ConformerBlock(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.first_norm = nn.LayerNorm(dimension)
        self.first_ffn = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dimension, dimension),
            nn.Dropout(0.1),
        )
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = ExportableSelfAttention(dimension, heads, dropout=0.1)
        self.convolution = ConformerConvolution(dimension)
        self.second_norm = nn.LayerNorm(dimension)
        self.second_ffn = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dimension, dimension),
            nn.Dropout(0.1),
        )
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        output = inputs + 0.5 * self.first_ffn(self.first_norm(inputs))
        normalized = self.attention_norm(output)
        attended = self.attention(normalized, padding_mask)
        output = output + attended
        output = output + self.convolution(output)
        output = output + 0.5 * self.second_ffn(self.second_norm(output))
        return self.output_norm(output)


def sinusoidal_positions(steps: int, dimension: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(steps, device=device, dtype=torch.float32).unsqueeze(1)
    divisors = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dimension)
    )
    encoding = torch.zeros(steps, dimension, device=device)
    encoding[:, 0::2] = torch.sin(positions * divisors)
    encoding[:, 1::2] = torch.cos(positions * divisors[: encoding[:, 1::2].shape[1]])
    return encoding


class TinyConformerEncoder(nn.Module):
    output_size = 96

    def __init__(self, n_mels: int) -> None:
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv1d(n_mels, 96, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.SiLU(),
            nn.Conv1d(96, 96, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ConformerBlock(96, 4) for _ in range(4)])

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.subsample(features.squeeze(1)).transpose(1, 2)
        lengths = convolution_length(lengths, 5, 2, 2)
        lengths = convolution_length(lengths, 5, 2, 2)
        output = output + sinusoidal_positions(
            output.shape[1], output.shape[2], output.device
        ).unsqueeze(0)
        padding_mask = ~sequence_mask(lengths, output.shape[1])
        for block in self.blocks:
            output = block(output, padding_mask)
        return output, lengths


class AttentionPool(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.score = nn.Linear(dimension, 1)

    def forward(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = sequence_mask(lengths, sequence.shape[1])
        scores = self.score(sequence).squeeze(-1).masked_fill(~mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class VoiceCommandModel(nn.Module):
    def __init__(
        self,
        architecture: str,
        rejection_strategy: str,
        audio_config: dict[str, Any],
        intent_count: int = 19,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.rejection_strategy = rejection_strategy
        self.intent_count = intent_count
        self.frontend = LogMelFrontend(audio_config)
        encoders: dict[str, type[nn.Module]] = {
            "dscnn": DSCNNEncoder,
            "tcresnet": TCResNetEncoder,
            "bcresnet": BCResNetEncoder,
            "crnn_attention": CRNNAttentionEncoder,
            "tiny_conformer": TinyConformerEncoder,
        }
        if architecture not in encoders:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.encoder = encoders[architecture](int(audio_config["n_mels"]))
        dimension = int(self.encoder.output_size)
        self.pool = AttentionPool(dimension)
        output_intents = intent_count + (1 if rejection_strategy == "unknown_class" else 0)
        self.intent_head = nn.Linear(dimension, output_intents)
        self.slot_head = nn.Linear(dimension, len(SLOT_TOKENS))
        self.scope_head = nn.Linear(dimension, 1) if rejection_strategy == "binary_scope" else None

    def forward(
        self, waveform: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, frame_lengths = self.frontend(waveform, lengths)
        sequence, sequence_lengths = self.encoder(features, frame_lengths)
        pooled = self.pool(sequence, sequence_lengths)
        intent_logits = self.intent_head(pooled)
        slot_logits = self.slot_head(sequence)
        if self.scope_head is None:
            scope_logits = torch.zeros(
                waveform.shape[0], 1, dtype=waveform.dtype, device=waveform.device
            )
        else:
            scope_logits = self.scope_head(pooled)
        return intent_logits, slot_logits, scope_logits, sequence_lengths


def build_model(
    architecture: str,
    rejection_strategy: str,
    audio_config: dict[str, Any],
) -> VoiceCommandModel:
    if rejection_strategy not in {"confidence", "unknown_class", "binary_scope"}:
        raise ValueError(f"Unknown rejection strategy: {rejection_strategy}")
    return VoiceCommandModel(architecture, rejection_strategy, audio_config)

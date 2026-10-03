from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PowerRequest(StrictRequest):
    on: StrictBool


class BrightnessRequest(StrictRequest):
    percent: float

    @field_validator("percent", mode="before")
    @classmethod
    def validate_percent(cls, value: Any) -> float:
        number = _finite_number(value, "percent")
        if not 0 <= number <= 100:
            raise ValueError("percent must be between 0 and 100")
        return number


class ColorRequest(StrictRequest):
    color: str

    @field_validator("color")
    @classmethod
    def validate_color(cls, value: str) -> str:
        normalized = " ".join(value.lower().split())
        if not normalized:
            raise ValueError("color cannot be empty")
        if len(normalized) > 64:
            raise ValueError("color cannot exceed 64 characters")
        return normalized


class TemperatureRequest(StrictRequest):
    degrees: float
    unit: Literal["celsius", "fahrenheit"]

    @field_validator("degrees", mode="before")
    @classmethod
    def validate_degrees(cls, value: Any) -> float:
        return _finite_number(value, "degrees")


class MediaStatusRequest(StrictRequest):
    connected: StrictBool
    playback_state: Literal["STOPPED", "PLAYING", "PAUSED"]
    device_id: str | None = Field(default=None, max_length=256)
    track_name: str | None = Field(default=None, max_length=256)
    artist_name: str | None = Field(default=None, max_length=256)
    position_ms: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)
    volume_percent: float
    error: str | None = Field(default=None, max_length=500)
    command_id: str | None = Field(default=None, max_length=128)

    @field_validator("volume_percent", mode="before")
    @classmethod
    def validate_volume(cls, value: Any) -> float:
        number = _finite_number(value, "volume_percent")
        if not 0 <= number <= 100:
            raise ValueError("volume_percent must be between 0 and 100")
        return number


class MediaVolumeRequest(StrictRequest):
    percent: float

    @field_validator("percent", mode="before")
    @classmethod
    def validate_percent(cls, value: Any) -> float:
        number = _finite_number(value, "percent")
        if not 0 <= number <= 100:
            raise ValueError("percent must be between 0 and 100")
        return number


class MutationResponse(BaseModel):
    status: Literal["success"] = "success"
    changed: bool
    request_id: str | None
    message: str
    state: dict[str, Any]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ErrorDetail

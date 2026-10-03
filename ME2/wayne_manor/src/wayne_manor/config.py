from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APPLICATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = APPLICATION_ROOT / "config" / "default.json"
DEVICE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True)
class ServerConfig:
    host: str
    port: int
    maximum_request_bytes: int


@dataclass(frozen=True)
class LightConfig:
    display_name: str
    initial_power: bool
    initial_brightness_percent: float
    initial_color: str
    supported_colors: dict[str, str]


@dataclass(frozen=True)
class ThermostatConfig:
    display_name: str
    unit: str
    initial_setpoint: float
    minimum_setpoint: float
    maximum_setpoint: float
    change_sound_url: str | None
    change_sound_volume: float


@dataclass(frozen=True)
class TelephoneConfig:
    display_name: str
    ring_duration_seconds: float
    ringtone_url: str | None


@dataclass(frozen=True)
class SpotifyConfig:
    client_id: str
    redirect_uri: str
    device_name: str
    default_uri: str
    initial_volume: float
    command_timeout_seconds: float
    status_timeout_seconds: float


@dataclass(frozen=True)
class WayneManorConfig:
    path: Path
    server: ServerConfig
    default_light_group: str
    default_thermostat: str
    lights: dict[str, LightConfig]
    thermostats: dict[str, ThermostatConfig]
    telephone: TelephoneConfig
    spotify: SpotifyConfig


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _device_id(value: Any, label: str) -> str:
    device_id = _text(value, label)
    if DEVICE_ID.fullmatch(device_id) is None:
        raise ValueError(f"{label} must use lowercase hyphenated identifiers")
    return device_id


def _local_asset_url(value: Any, label: str) -> str | None:
    if value is None:
        return None
    url = _text(value, label)
    if not url.startswith("/assets/") or ".." in url:
        raise ValueError(f"{label} must be a local /assets/ path")
    return url


def _unit_interval(value: Any, label: str) -> float:
    number = _finite(value, label)
    if not 0 <= number <= 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return number


def load_config(path: Path | None = None) -> WayneManorConfig:
    config_path = (path or DEFAULT_CONFIG).expanduser().resolve()
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Wayne Manor configuration is missing: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid Wayne Manor configuration: {error}") from error
    document = _object(document, "configuration")

    server_values = _object(document.get("server"), "server")
    host = _text(server_values.get("host"), "server.host")
    port = int(server_values.get("port", 0))
    maximum_request_bytes = int(server_values.get("maximum_request_bytes", 0))
    if not 1 <= port <= 65_535:
        raise ValueError("server.port must be between 1 and 65535")
    if not 1_024 <= maximum_request_bytes <= 1_048_576:
        raise ValueError("server.maximum_request_bytes must be between 1024 and 1048576")

    light_values = _object(document.get("lights"), "lights")
    if not light_values:
        raise ValueError("At least one light group is required")
    lights: dict[str, LightConfig] = {}
    for raw_id, raw in light_values.items():
        light_id = _device_id(raw_id, "light group ID")
        values = _object(raw, f"lights.{light_id}")
        brightness = _finite(
            values.get("initial_brightness_percent"),
            f"lights.{light_id}.initial_brightness_percent",
        )
        if not 0 <= brightness <= 100:
            raise ValueError(f"lights.{light_id}.initial_brightness_percent must be 0–100")
        colors_value = _object(
            values.get("supported_colors"), f"lights.{light_id}.supported_colors"
        )
        if not colors_value:
            raise ValueError(f"lights.{light_id} requires at least one supported color")
        colors: dict[str, str] = {}
        for raw_name, raw_hex in colors_value.items():
            name = " ".join(_text(raw_name, "color name").lower().split())
            color_hex = _text(raw_hex, f"color {name}").lower()
            if HEX_COLOR.fullmatch(color_hex) is None:
                raise ValueError(f"Color {name!r} must use #RRGGBB")
            if name in colors:
                raise ValueError(f"Duplicate normalized color name: {name}")
            colors[name] = color_hex
        initial_color = " ".join(
            _text(values.get("initial_color"), f"lights.{light_id}.initial_color")
            .lower()
            .split()
        )
        if initial_color not in colors:
            raise ValueError(f"lights.{light_id}.initial_color is not supported")
        initial_power = values.get("initial_power")
        if not isinstance(initial_power, bool):
            raise TypeError(f"lights.{light_id}.initial_power must be true or false")
        lights[light_id] = LightConfig(
            display_name=_text(values.get("display_name"), f"lights.{light_id}.display_name"),
            initial_power=initial_power,
            initial_brightness_percent=brightness,
            initial_color=initial_color,
            supported_colors=colors,
        )

    thermostat_values = _object(document.get("thermostats"), "thermostats")
    if not thermostat_values:
        raise ValueError("At least one thermostat is required")
    thermostats: dict[str, ThermostatConfig] = {}
    for raw_id, raw in thermostat_values.items():
        thermostat_id = _device_id(raw_id, "thermostat ID")
        values = _object(raw, f"thermostats.{thermostat_id}")
        unit = _text(values.get("unit"), f"thermostats.{thermostat_id}.unit").lower()
        if unit not in {"celsius", "fahrenheit"}:
            raise ValueError(f"thermostats.{thermostat_id}.unit must be celsius or fahrenheit")
        minimum = _finite(
            values.get("minimum_setpoint"), f"thermostats.{thermostat_id}.minimum_setpoint"
        )
        maximum = _finite(
            values.get("maximum_setpoint"), f"thermostats.{thermostat_id}.maximum_setpoint"
        )
        initial = _finite(
            values.get("initial_setpoint"), f"thermostats.{thermostat_id}.initial_setpoint"
        )
        if minimum >= maximum:
            raise ValueError(f"thermostats.{thermostat_id} minimum must be below maximum")
        if not minimum <= initial <= maximum:
            raise ValueError(f"thermostats.{thermostat_id} initial setpoint is outside its limits")
        thermostats[thermostat_id] = ThermostatConfig(
            display_name=_text(
                values.get("display_name"), f"thermostats.{thermostat_id}.display_name"
            ),
            unit=unit,
            initial_setpoint=initial,
            minimum_setpoint=minimum,
            maximum_setpoint=maximum,
            change_sound_url=_local_asset_url(
                values.get("change_sound_url"),
                f"thermostats.{thermostat_id}.change_sound_url",
            ),
            change_sound_volume=_unit_interval(
                values.get("change_sound_volume"),
                f"thermostats.{thermostat_id}.change_sound_volume",
            ),
        )

    default_light = _device_id(document.get("default_light_group"), "default_light_group")
    default_thermostat = _device_id(
        document.get("default_thermostat"), "default_thermostat"
    )
    if default_light not in lights:
        raise ValueError("default_light_group does not identify a configured light group")
    if default_thermostat not in thermostats:
        raise ValueError("default_thermostat does not identify a configured thermostat")

    telephone_values = _object(document.get("telephone"), "telephone")
    ring_duration_seconds = _finite(
        telephone_values.get("ring_duration_seconds"), "telephone.ring_duration_seconds"
    )
    if not 1 <= ring_duration_seconds <= 60:
        raise ValueError("telephone.ring_duration_seconds must be between 1 and 60")
    ringtone_url = _local_asset_url(
        telephone_values.get("ringtone_url"), "telephone.ringtone_url"
    )
    telephone = TelephoneConfig(
        display_name=_text(telephone_values.get("display_name"), "telephone.display_name"),
        ring_duration_seconds=ring_duration_seconds,
        ringtone_url=ringtone_url,
    )

    spotify_values = _object(document.get("spotify"), "spotify")
    client_id = _text(spotify_values.get("client_id"), "spotify.client_id")
    if re.fullmatch(r"[0-9a-fA-F]{32}", client_id) is None:
        raise ValueError("spotify.client_id must be a 32-character Spotify client ID")
    redirect_uri = _text(spotify_values.get("redirect_uri"), "spotify.redirect_uri")
    if not redirect_uri.startswith(("http://127.0.0.1:", "https://")):
        raise ValueError(
            "spotify.redirect_uri must use HTTPS or an HTTP 127.0.0.1 loopback address"
        )
    default_uri = _text(spotify_values.get("default_uri"), "spotify.default_uri")
    if not default_uri.startswith(
        ("spotify:track:", "spotify:album:", "spotify:artist:", "spotify:playlist:")
    ):
        raise ValueError("spotify.default_uri must be a Spotify playback URI")
    initial_volume = _finite(spotify_values.get("initial_volume"), "spotify.initial_volume")
    if not 0 <= initial_volume <= 1:
        raise ValueError("spotify.initial_volume must be between 0 and 1")
    command_timeout = _finite(
        spotify_values.get("command_timeout_seconds"), "spotify.command_timeout_seconds"
    )
    status_timeout = _finite(
        spotify_values.get("status_timeout_seconds"), "spotify.status_timeout_seconds"
    )
    if not 1 <= command_timeout <= 30:
        raise ValueError("spotify.command_timeout_seconds must be between 1 and 30")
    if not 5 <= status_timeout <= 60:
        raise ValueError("spotify.status_timeout_seconds must be between 5 and 60")
    spotify = SpotifyConfig(
        client_id=client_id,
        redirect_uri=redirect_uri,
        device_name=_text(spotify_values.get("device_name"), "spotify.device_name"),
        default_uri=default_uri,
        initial_volume=initial_volume,
        command_timeout_seconds=command_timeout,
        status_timeout_seconds=status_timeout,
    )

    return WayneManorConfig(
        path=config_path,
        server=ServerConfig(host, port, maximum_request_bytes),
        default_light_group=default_light,
        default_thermostat=default_thermostat,
        lights=lights,
        thermostats=thermostats,
        telephone=telephone,
        spotify=spotify,
    )

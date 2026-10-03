from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
MAX_RESPONSE_BYTES = 256_000

WEATHER_DESCRIPTIONS = {
    0: "clear",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "foggy with rime",
    51: "lightly drizzling",
    53: "drizzling",
    55: "heavily drizzling",
    56: "lightly freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snowing",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "heavy rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "a thunderstorm",
    96: "a thunderstorm with light hail",
    99: "a thunderstorm with heavy hail",
}


class WeatherError(RuntimeError):
    """Expected weather-service failure with a safe user-facing message."""


@dataclass(frozen=True)
class CurrentWeather:
    temperature: float
    apparent_temperature: float
    humidity: float
    weather_code: int


@dataclass(frozen=True)
class WeatherLocation:
    name: str
    latitude: float
    longitude: float


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WeatherError(f"Open-Meteo returned an invalid {label}.")
    number = float(value)
    if not math.isfinite(number):
        raise WeatherError(f"Open-Meteo returned an invalid {label}.")
    return number


def _spoken_number(value: float) -> str:
    rounded = round(value)
    return str(rounded) if abs(value - rounded) < 0.05 else f"{value:.1f}"


class OpenMeteoClient:
    def __init__(
        self,
        *,
        location_name: str,
        latitude: float,
        longitude: float,
        temperature_unit: str,
        timeout_seconds: float,
        cache_seconds: float,
        maximum_location_characters: int = 100,
        opener: Callable[..., Any] = urllib.request.urlopen,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not location_name.strip():
            raise ValueError("Weather location name cannot be empty")
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError("Weather coordinates are outside their valid ranges")
        if temperature_unit not in {"celsius", "fahrenheit"}:
            raise ValueError("Weather temperature unit must be celsius or fahrenheit")
        if timeout_seconds <= 0 or cache_seconds < 0:
            raise ValueError("Weather timeout must be positive and cache duration non-negative")
        if not 2 <= maximum_location_characters <= 200:
            raise ValueError("Weather location limit must be between 2 and 200 characters")
        self.default_location = WeatherLocation(location_name.strip(), latitude, longitude)
        self.temperature_unit = temperature_unit
        self.timeout_seconds = timeout_seconds
        self.cache_seconds = cache_seconds
        self.maximum_location_characters = maximum_location_characters
        self.opener = opener
        self.monotonic = monotonic
        self._cache: dict[tuple[float, float], tuple[CurrentWeather, float]] = {}

    def _forecast_url(self, location: WeatherLocation) -> str:
        query = urllib.parse.urlencode(
            {
                "latitude": location.latitude,
                "longitude": location.longitude,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code"
                ),
                "temperature_unit": self.temperature_unit,
                "timezone": "auto",
            }
        )
        return f"{FORECAST_URL}?{query}"

    def _read_document(
        self,
        url: str,
        unavailable_message: str,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "Alfred/1.0"},
        )
        try:
            with self.opener(
                request,
                timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
            ) as response:
                payload = response.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            raise WeatherError(unavailable_message) from error
        if len(payload) > MAX_RESPONSE_BYTES:
            raise WeatherError("The weather service returned an unexpectedly large response.")
        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WeatherError("The weather service returned an invalid response.") from error
        if not isinstance(document, dict):
            raise WeatherError("The weather service returned an invalid response.")
        return document

    @staticmethod
    def _parse_current(document: dict[str, Any]) -> CurrentWeather:
        try:
            current = document["current"]
            weather = CurrentWeather(
                temperature=_number(current["temperature_2m"], "temperature"),
                apparent_temperature=_number(
                    current["apparent_temperature"], "apparent temperature"
                ),
                humidity=_number(current["relative_humidity_2m"], "humidity"),
                weather_code=int(current["weather_code"]),
            )
            if not 0 <= weather.humidity <= 100:
                raise WeatherError("Open-Meteo returned an invalid humidity.")
        except (KeyError, TypeError, ValueError) as error:
            raise WeatherError("The weather service returned an invalid response.") from error
        return weather

    def check_availability(self, timeout_seconds: float) -> bool:
        """Probe the configured default forecast without changing the cache."""
        try:
            document = self._read_document(
                self._forecast_url(self.default_location),
                "I'm unable to retrieve the weather right now.",
                timeout_seconds=timeout_seconds,
            )
            self._parse_current(document)
        except WeatherError:
            return False
        return True

    def resolve_location(self, query: str) -> WeatherLocation:
        normalized = " ".join(query.strip().split())
        if (
            len(normalized) < 2
            or len(normalized) > self.maximum_location_characters
            or any(ord(character) < 32 for character in normalized)
        ):
            raise WeatherError("I couldn't find that location.")
        url = f"{GEOCODING_URL}?{urllib.parse.urlencode({'name': normalized, 'count': 1, 'language': 'en', 'format': 'json'})}"
        document = self._read_document(url, "I couldn't find that location.")
        try:
            first = document["results"][0]
            name = str(first["name"]).strip()
            latitude = _number(first["latitude"], "location latitude")
            longitude = _number(first["longitude"], "location longitude")
        except (KeyError, IndexError, TypeError) as error:
            raise WeatherError("I couldn't find that location.") from error
        if not name or not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise WeatherError("I couldn't find that location.")
        return WeatherLocation(name, latitude, longitude)

    def current(self, location: WeatherLocation | None = None) -> CurrentWeather:
        selected = location or self.default_location
        cache_key = (selected.latitude, selected.longitude)
        now = self.monotonic()
        cached = self._cache.get(cache_key)
        if cached is not None and now - cached[1] <= self.cache_seconds:
            return cached[0]
        document = self._read_document(
            self._forecast_url(selected),
            "I'm unable to retrieve the weather right now.",
        )
        weather = self._parse_current(document)
        self._cache[cache_key] = (weather, now)
        return weather

    def spoken_summary(self, location: WeatherLocation | None = None) -> str:
        selected = location or self.default_location
        current = self.current(selected)
        condition = WEATHER_DESCRIPTIONS.get(current.weather_code, "unsettled weather")
        unit = "degrees Celsius" if self.temperature_unit == "celsius" else "degrees Fahrenheit"
        return (
            f"In {selected.name}, it is {condition} and "
            f"{_spoken_number(current.temperature)} {unit}. It feels like "
            f"{_spoken_number(current.apparent_temperature)} degrees, with "
            f"{_spoken_number(current.humidity)} percent humidity."
        )

    def spoken_summary_for(self, location_query: str | None) -> str:
        # A missing query means the caller intentionally selected the configured
        # default (for example, Alfred is running without clarification ASR).
        # It is not a failed location lookup and should not be announced as one.
        if not location_query:
            return self.spoken_summary()

        fallback_message = (
            f"I couldn't use that location, so I'll check {self.default_location.name}."
        )
        try:
            location = self.resolve_location(location_query)
            return self.spoken_summary(location)
        except WeatherError:
            try:
                return f"{fallback_message}\n{self.spoken_summary()}"
            except WeatherError:
                raise WeatherError("I'm unable to retrieve the weather right now.") from None

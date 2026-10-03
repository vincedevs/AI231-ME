from __future__ import annotations

import json
import urllib.error
import urllib.parse

import pytest

from alfred.weather import OpenMeteoClient, WeatherError


class Response:
    def __init__(self, document: dict) -> None:
        self.payload = json.dumps(document).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, limit: int) -> bytes:
        assert limit > len(self.payload)
        return self.payload


def test_open_meteo_uses_fixed_quezon_city_coordinates_and_caches_response() -> None:
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        return Response(
            {
                "current": {
                    "temperature_2m": 30.2,
                    "apparent_temperature": 34.0,
                    "relative_humidity_2m": 72,
                    "weather_code": 2,
                }
            }
        )

    client = OpenMeteoClient(
        location_name="Quezon City",
        latitude=14.6488,
        longitude=121.0509,
        temperature_unit="celsius",
        timeout_seconds=5,
        cache_seconds=600,
        opener=opener,
        monotonic=lambda: 100,
    )
    expected = (
        "In Quezon City, it is partly cloudy and 30.2 degrees Celsius. "
        "It feels like 34 degrees, with 72 percent humidity."
    )
    assert client.spoken_summary() == expected
    assert client.spoken_summary() == expected
    assert client.spoken_summary_for(None) == expected
    assert len(calls) == 1
    query = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0][0]).query)
    assert query["latitude"] == ["14.6488"]
    assert query["longitude"] == ["121.0509"]
    assert query["timezone"] == ["auto"]
    assert calls[0][1] == 5


def test_availability_check_uses_short_timeout_without_populating_cache() -> None:
    calls = []

    def opener(request, timeout):
        calls.append(timeout)
        return Response(
            {
                "current": {
                    "temperature_2m": 30,
                    "apparent_temperature": 34,
                    "relative_humidity_2m": 72,
                    "weather_code": 2,
                }
            }
        )

    client = OpenMeteoClient(
        location_name="Quezon City",
        latitude=14.6488,
        longitude=121.0509,
        temperature_unit="celsius",
        timeout_seconds=5,
        cache_seconds=600,
        opener=opener,
    )

    assert client.check_availability(0.25) is True
    client.current()
    assert calls == [0.25, 5]


def test_availability_check_rejects_unreachable_or_invalid_weather_service() -> None:
    unavailable = OpenMeteoClient(
        location_name="Quezon City",
        latitude=14.6488,
        longitude=121.0509,
        temperature_unit="celsius",
        timeout_seconds=5,
        cache_seconds=600,
        opener=lambda request, timeout: (_ for _ in ()).throw(urllib.error.URLError("offline")),
    )
    invalid = OpenMeteoClient(
        location_name="Quezon City",
        latitude=14.6488,
        longitude=121.0509,
        temperature_unit="celsius",
        timeout_seconds=5,
        cache_seconds=600,
        opener=lambda request, timeout: Response({"current": {}}),
    )

    assert unavailable.check_availability(0.25) is False
    assert invalid.check_availability(0.25) is False


def test_open_meteo_resolves_requested_location_before_fetching_weather() -> None:
    calls = []

    def opener(request, timeout):
        calls.append((request.full_url, timeout))
        if request.full_url.startswith("https://geocoding-api.open-meteo.com"):
            return Response(
                {
                    "results": [
                        {
                            "name": "Makati",
                            "latitude": 14.5547,
                            "longitude": 121.0244,
                        }
                    ]
                }
            )
        return Response(
            {
                "current": {
                    "temperature_2m": 29,
                    "apparent_temperature": 32,
                    "relative_humidity_2m": 70,
                    "weather_code": 1,
                }
            }
        )

    client = OpenMeteoClient(
        location_name="Quezon City",
        latitude=14.6488,
        longitude=121.0509,
        temperature_unit="celsius",
        timeout_seconds=5,
        cache_seconds=600,
        opener=opener,
    )

    assert client.spoken_summary_for("Makati City") == (
        "In Makati, it is mainly clear and 29 degrees Celsius. "
        "It feels like 32 degrees, with 70 percent humidity."
    )
    assert len(calls) == 2
    geocoding_query = urllib.parse.parse_qs(urllib.parse.urlparse(calls[0][0]).query)
    assert geocoding_query == {
        "name": ["Makati City"],
        "count": ["1"],
        "language": ["en"],
        "format": ["json"],
    }
    forecast_query = urllib.parse.parse_qs(urllib.parse.urlparse(calls[1][0]).query)
    assert forecast_query["latitude"] == ["14.5547"]
    assert forecast_query["longitude"] == ["121.0244"]


def test_unknown_location_falls_back_to_quezon_city() -> None:
    calls = []

    def opener(request, timeout):
        del timeout
        calls.append(request.full_url)
        if request.full_url.startswith("https://geocoding-api.open-meteo.com"):
            return Response({"results": []})
        return Response(
            {
                "current": {
                    "temperature_2m": 30,
                    "apparent_temperature": 34,
                    "relative_humidity_2m": 72,
                    "weather_code": 2,
                }
            }
        )

    client = OpenMeteoClient(
        location_name="Quezon City",
        latitude=14.6488,
        longitude=121.0509,
        temperature_unit="celsius",
        timeout_seconds=5,
        cache_seconds=600,
        opener=opener,
    )

    assert client.spoken_summary_for("not a real location") == (
        "I couldn't use that location, so I'll check Quezon City.\n"
        "In Quezon City, it is partly cloudy and 30 degrees Celsius. "
        "It feels like 34 degrees, with 72 percent humidity."
    )
    assert len(calls) == 2
    fallback_query = urllib.parse.parse_qs(urllib.parse.urlparse(calls[1]).query)
    assert fallback_query["latitude"] == ["14.6488"]
    assert fallback_query["longitude"] == ["121.0509"]


def test_requested_forecast_failure_falls_back_to_quezon_city() -> None:
    calls = []

    def opener(request, timeout):
        del timeout
        calls.append(request.full_url)
        if request.full_url.startswith("https://geocoding-api.open-meteo.com"):
            return Response(
                {"results": [{"name": "Makati", "latitude": 14.5547, "longitude": 121.0244}]}
            )
        query = urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        if query["latitude"] == ["14.5547"]:
            raise urllib.error.URLError("fixture outage")
        return Response(
            {
                "current": {
                    "temperature_2m": 30,
                    "apparent_temperature": 34,
                    "relative_humidity_2m": 72,
                    "weather_code": 2,
                }
            }
        )

    client = OpenMeteoClient(
        location_name="Quezon City",
        latitude=14.6488,
        longitude=121.0509,
        temperature_unit="celsius",
        timeout_seconds=5,
        cache_seconds=600,
        opener=opener,
    )

    result = client.spoken_summary_for("Makati City")
    assert result.startswith("I couldn't use that location, so I'll check Quezon City.\n")
    assert len(calls) == 3


@pytest.mark.parametrize("humidity", [-1, 101, float("nan")])
def test_open_meteo_rejects_invalid_measurements(humidity: float) -> None:
    client = OpenMeteoClient(
        location_name="Quezon City",
        latitude=14.6488,
        longitude=121.0509,
        temperature_unit="celsius",
        timeout_seconds=5,
        cache_seconds=0,
        opener=lambda request, timeout: Response(
            {
                "current": {
                    "temperature_2m": 30,
                    "apparent_temperature": 34,
                    "relative_humidity_2m": humidity,
                    "weather_code": 2,
                }
            }
        ),
    )
    with pytest.raises(WeatherError):
        client.current()

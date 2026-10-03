from __future__ import annotations

import io
import json
import urllib.error

import pytest

from alfred.wayne_manor import WayneManorClient, WayneManorError


class Response:
    def __init__(self, document: dict) -> None:
        self.payload = json.dumps(document).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


def client(opener) -> WayneManorClient:
    return WayneManorClient(
        base_url="http://127.0.0.1:8765/api/v1",
        light_group="living-room",
        thermostat="living-room-climate",
        temperature_unit="celsius",
        timeout_seconds=2.0,
        opener=opener,
    )


def test_client_sends_json_and_correlates_request_id() -> None:
    captured = []

    def opener(request, timeout):
        captured.append((request, timeout))
        return Response({"status": "success", "request_id": "req-1", "message": "Done."})

    result = client(opener).set_brightness(65.0, "req-1")
    request, timeout = captured[0]
    assert result == "Done."
    assert request.full_url.endswith("/light-groups/living-room/brightness")
    assert json.loads(request.data) == {"percent": 65.0}
    assert request.get_header("X-request-id") == "req-1"
    assert timeout == 2.0


def test_client_maps_all_device_operations_to_the_api_contract() -> None:
    captured = []

    def opener(request, timeout):
        del timeout
        captured.append((request.full_url, json.loads(request.data)))
        request_id = request.get_header("X-request-id")
        return Response({"status": "success", "request_id": request_id, "message": "Done."})

    wayne_manor = client(opener)
    wayne_manor.set_light_power(True, "power-on")
    wayne_manor.set_light_power(False, "power-off")
    wayne_manor.set_color("warm white", "color")
    wayne_manor.set_temperature(22.5, "temperature")
    wayne_manor.call("call")
    assert [(url.rsplit("/api/v1/", 1)[1], body) for url, body in captured] == [
        ("light-groups/living-room/power", {"on": True}),
        ("light-groups/living-room/power", {"on": False}),
        ("light-groups/living-room/color", {"color": "warm white"}),
        (
            "thermostats/living-room-climate/setpoint",
            {"degrees": 22.5, "unit": "celsius"},
        ),
        ("telephone/calls", {}),
    ]
    assert captured[-1][0].endswith("/telephone/calls")


def test_client_preserves_safe_api_error_message() -> None:
    payload = json.dumps(
        {"error": {"code": "unsupported_color", "message": "Cerise is not supported."}}
    ).encode("utf-8")

    def opener(request, timeout):
        del timeout
        raise urllib.error.HTTPError(request.full_url, 422, "", {}, io.BytesIO(payload))

    with pytest.raises(WayneManorError, match="Cerise is not supported"):
        client(opener).set_color("cerise", "req-2")


def test_client_rejects_uncorrelated_or_invalid_responses() -> None:
    def wrong_request_id(request, timeout):
        del request, timeout
        return Response({"status": "success", "request_id": "wrong", "message": "Done."})

    with pytest.raises(WayneManorError, match="invalid response"):
        client(wrong_request_id).set_light_power(True, "req-3")


def test_client_reports_connection_failure_without_internal_details() -> None:
    def unavailable(request, timeout):
        del request, timeout
        raise urllib.error.URLError("connection refused")

    with pytest.raises(WayneManorError, match="unavailable right now"):
        client(unavailable).set_temperature(24.0, "req-4")


def test_health_check_is_read_only_and_uses_startup_timeout() -> None:
    captured = []

    def opener(request, timeout):
        captured.append((request.full_url, request.method, timeout))
        return Response({"status": "ok"})

    assert client(opener).check_availability(0.25) is True
    assert captured == [("http://127.0.0.1:8765/api/v1/health", "GET", 0.25)]


def test_health_check_returns_false_when_wayne_manor_is_unreachable() -> None:
    def unavailable(request, timeout):
        del request, timeout
        raise urllib.error.URLError("offline")

    assert client(unavailable).check_availability(0.25) is False


def test_client_reads_media_state_and_sends_correlated_commands() -> None:
    captured = []

    def opener(request, timeout):
        del timeout
        captured.append((request.full_url, request.method, request.data))
        if request.method == "GET":
            return Response({"connected": True, "playback_state": "STOPPED"})
        request_id = request.get_header("X-request-id")
        return Response({"status": "success", "request_id": request_id, "message": "Music paused."})

    wayne_manor = client(opener)
    assert wayne_manor.media_state() == {
        "connected": True,
        "playback_state": "STOPPED",
    }
    assert wayne_manor.media_command("pause", "media-1") == "Music paused."
    assert wayne_manor.set_media_volume(65, "volume-1") == "Music paused."
    assert captured == [
        ("http://127.0.0.1:8765/api/v1/media", "GET", None),
        ("http://127.0.0.1:8765/api/v1/media/pause", "POST", b"{}"),
        ("http://127.0.0.1:8765/api/v1/media/volume", "PUT", b'{"percent":65}'),
    ]

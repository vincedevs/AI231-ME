from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from wayne_manor.app import create_app
from wayne_manor.config import load_config


def client() -> TestClient:
    return TestClient(create_app(load_config()))


def test_health_capabilities_and_initial_state() -> None:
    with client() as test_client:
        assert test_client.get("/api/v1/health").json() == {
            "status": "ok",
        }
        capabilities = test_client.get("/api/v1/capabilities").json()
        assert capabilities["default_light_group"] == "living-room"
        assert "blue" in capabilities["lights"]["living-room"]["supported_colors"]
        state = test_client.get("/api/v1/state").json()
        assert state["lights"]["living-room"]["effective_intensity"] == 0.65
        assert state["telephone"]["ringing"] is False
        assert capabilities["telephone"]["ringtone_url"] == "/assets/telephone-ring.wav"
        thermostat = capabilities["thermostats"]["living-room-climate"]
        assert thermostat["change_sound_url"] == "/assets/air-conditioner.wav"
        assert thermostat["change_sound_volume"] == 0.5
        assert capabilities["spotify"]["client_id"] == "3fcb96d2b6a84fbdbffff892713e83ca"
        assert state["media"]["connected"] is False
        assert state["media"]["playback_state"] == "STOPPED"
        assert state["media"]["volume_percent"] == 30
        assert capabilities["spotify"]["initial_volume"] == 0.3
        assert "revision" not in state
        assert test_client.get("/api/v1/ui-config").json() == {"mode": "display"}


def test_ui_mode_is_explicit_and_validated() -> None:
    with TestClient(create_app(load_config(), ui_mode="controls")) as test_client:
        assert test_client.get("/api/v1/ui-config").json() == {"mode": "controls"}
    with pytest.raises(ValueError, match="ui_mode must be display or controls"):
        create_app(load_config(), ui_mode="unknown")


def test_mutations_are_validated_and_echo_request_id() -> None:
    with client() as test_client:
        response = test_client.put(
            "/api/v1/light-groups/living-room/brightness",
            json={"percent": 42},
            headers={"X-Request-ID": "request-1"},
        )
        assert response.status_code == 200
        assert response.json()["request_id"] == "request-1"
        assert response.json()["state"]["brightness_percent"] == 42

        invalid = test_client.put(
            "/api/v1/light-groups/living-room/brightness", json={"percent": 101}
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "invalid_request"

        unsupported = test_client.put(
            "/api/v1/light-groups/living-room/color", json={"color": "infrared"}
        )
        assert unsupported.status_code == 422
        assert unsupported.json()["error"]["code"] == "unsupported_color"

        state = test_client.get("/api/v1/state").json()
        assert state["lights"]["living-room"]["brightness_percent"] == 42


def test_temperature_and_missing_device_errors_have_stable_envelopes() -> None:
    with client() as test_client:
        response = test_client.put(
            "/api/v1/thermostats/living-room-climate/setpoint",
            json={"degrees": 22, "unit": "celsius"},
        )
        assert response.status_code == 200
        assert response.json()["state"]["setpoint"] == 22

        unsafe = test_client.put(
            "/api/v1/thermostats/living-room-climate/setpoint",
            json={"degrees": 40, "unit": "celsius"},
        )
        assert unsafe.status_code == 422
        assert unsafe.json()["error"]["code"] == "unsafe_temperature"

        missing = test_client.get("/api/v1/light-groups/unknown")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "light_group_not_found"


def test_openapi_contains_every_mutation_endpoint() -> None:
    with client() as test_client:
        paths = test_client.get("/openapi.json").json()["paths"]
    assert "/api/v1/light-groups/{group_id}/power" in paths
    assert "/api/v1/light-groups/{group_id}/brightness" in paths
    assert "/api/v1/light-groups/{group_id}/color" in paths
    assert "/api/v1/thermostats/{thermostat_id}/setpoint" in paths
    assert "/api/v1/ui-config" in paths
    assert "/api/v1/telephone/calls" in paths
    assert "/api/v1/telephone/calls/current" in paths
    assert "/api/v1/media/play" in paths
    assert "/api/v1/media/status" in paths
    assert "/api/v1/media/volume" in paths


def test_telephone_call_rings_and_can_be_stopped() -> None:
    with client() as test_client:
        started = test_client.post(
            "/api/v1/telephone/calls",
            headers={"X-Request-ID": "call-1"},
        )
        assert started.status_code == 200
        assert started.json()["request_id"] == "call-1"
        assert started.json()["state"]["ringing"] is True
        assert test_client.get("/api/v1/state").json()["telephone"]["call_id"] == "call-1"

        stopped = test_client.delete(
            "/api/v1/telephone/calls/current",
            headers={"X-Request-ID": "stop-1"},
        )
        assert stopped.status_code == 200
        assert stopped.json()["state"]["ringing"] is False


def test_malformed_strict_and_oversized_requests_are_rejected() -> None:
    with client() as test_client:
        malformed = test_client.put(
            "/api/v1/light-groups/living-room/color",
            content=b"{not-json",
            headers={"Content-Type": "application/json"},
        )
        assert malformed.status_code == 422
        assert malformed.json()["error"]["code"] == "invalid_request"

        coerced_boolean = test_client.put(
            "/api/v1/light-groups/living-room/power", json={"on": "yes"}
        )
        assert coerced_boolean.status_code == 422

        oversized = test_client.put(
            "/api/v1/light-groups/living-room/color",
            content=b"x" * 16_385,
            headers={"Content-Type": "application/json"},
        )
        assert oversized.status_code == 413
        assert oversized.json()["error"]["code"] == "request_too_large"


def test_media_command_explains_when_browser_player_is_not_connected() -> None:
    with client() as test_client:
        response = test_client.post(
            "/api/v1/media/play", headers={"X-Request-ID": "media-api-1"}
        )
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "spotify_not_connected"

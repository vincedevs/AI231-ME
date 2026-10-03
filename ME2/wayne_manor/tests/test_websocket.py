from __future__ import annotations

from fastapi.testclient import TestClient

from wayne_manor.app import create_app
from wayne_manor.config import load_config


def test_websocket_sends_snapshot_then_only_changed_state() -> None:
    with (
        TestClient(create_app(load_config())) as client,
        client.websocket_connect("/api/v1/events") as socket,
    ):
        initial = socket.receive_json()
        assert initial["type"] == "state.snapshot"
        assert "revision" not in initial

        response = client.put(
            "/api/v1/light-groups/living-room/power", json={"on": False}
        )
        assert response.status_code == 200
        event = socket.receive_json()
        assert event["type"] == "state.changed"
        assert event["state"]["lights"]["living-room"]["on"] is False


def test_multiple_websocket_clients_receive_the_same_state_change() -> None:
    with (
        TestClient(create_app(load_config())) as client,
        client.websocket_connect("/api/v1/events") as first,
        client.websocket_connect("/api/v1/events") as second,
    ):
        assert first.receive_json()["type"] == "state.snapshot"
        assert second.receive_json()["type"] == "state.snapshot"
        response = client.put(
            "/api/v1/light-groups/living-room/color", json={"color": "blue"}
        )
        assert response.status_code == 200
        assert first.receive_json()["state"]["lights"]["living-room"]["color"] == "blue"
        assert second.receive_json()["state"]["lights"]["living-room"]["color"] == "blue"

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from wayne_manor.config import load_config
from wayne_manor.state import WayneManorApiError, WorldStore


def test_light_state_is_idempotent_and_preserves_settings() -> None:
    async def scenario() -> None:
        store = WorldStore(load_config())
        initial = await store.light("living-room")
        assert initial["on"] is True
        assert initial["brightness_percent"] == 65

        off = await store.set_light_power("living-room", False)
        assert off.changed is True
        assert off.state["effective_intensity"] == 0

        duplicate = await store.set_light_power("living-room", False)
        assert duplicate.changed is False

        await store.set_light_brightness("living-room", 35.0)
        await store.set_light_color("living-room", "blue")
        on = await store.set_light_power("living-room", True)
        assert on.state["brightness_percent"] == 35
        assert on.state["color"] == "blue"
        assert on.state["effective_intensity"] == 0.35

    asyncio.run(scenario())


def test_color_and_temperature_capabilities_are_enforced() -> None:
    async def scenario() -> None:
        store = WorldStore(load_config())
        with pytest.raises(WayneManorApiError) as color_error:
            await store.set_light_color("living-room", "infrared")
        assert color_error.value.code == "unsupported_color"

        for degrees in (16.0, 30.0):
            outcome = await store.set_temperature("living-room-climate", degrees, "celsius")
            assert outcome.state["setpoint"] == degrees
        with pytest.raises(WayneManorApiError) as temperature_error:
            await store.set_temperature("living-room-climate", 30.1, "celsius")
        assert temperature_error.value.code == "unsafe_temperature"
        with pytest.raises(WayneManorApiError) as unit_error:
            await store.set_temperature("living-room-climate", 22, "fahrenheit")
        assert unit_error.value.code == "temperature_unit_mismatch"

    asyncio.run(scenario())


def test_concurrent_changes_leave_a_complete_state_snapshot() -> None:
    async def scenario() -> None:
        store = WorldStore(load_config())
        outcomes = await asyncio.gather(
            store.set_light_brightness("living-room", 10.0),
            store.set_light_brightness("living-room", 20.0),
            store.set_light_brightness("living-room", 30.0),
        )
        assert all(outcome.changed for outcome in outcomes)
        snapshot = await store.snapshot()
        assert snapshot["lights"]["living-room"]["brightness_percent"] in {10, 20, 30}
        snapshot["lights"]["living-room"]["color"] = "tampered"
        assert (await store.light("living-room"))["color"] == "warm white"

    asyncio.run(scenario())


def test_telephone_call_updates_state_and_stops_explicitly() -> None:
    async def scenario() -> None:
        store = WorldStore(load_config())
        started = await store.start_telephone_call("call-1")
        assert started.changed is True
        assert started.state["ringing"] is True
        assert started.state["call_id"] == "call-1"

        stopped = await store.stop_telephone_call()
        assert stopped.changed is True
        assert stopped.state["ringing"] is False
        assert (await store.snapshot())["telephone"]["ringing"] is False
        await store.close()

    asyncio.run(scenario())


def test_telephone_call_stops_after_configured_duration() -> None:
    async def scenario() -> None:
        config = load_config()
        fast_config = replace(
            config,
            telephone=replace(config.telephone, ring_duration_seconds=0.01),
        )
        store = WorldStore(fast_config)
        await store.start_telephone_call("call-2")
        await asyncio.sleep(0.03)
        snapshot = await store.snapshot()
        assert snapshot["telephone"]["ringing"] is False
        assert snapshot["telephone"]["call_id"] is None
        await store.close()

    asyncio.run(scenario())


def test_media_command_waits_for_matching_browser_acknowledgement() -> None:
    async def scenario() -> None:
        store = WorldStore(load_config())
        status = {
            "connected": True,
            "playback_state": "STOPPED",
            "device_id": "browser-device",
            "track_name": None,
            "artist_name": None,
            "position_ms": 0,
            "duration_ms": 0,
            "volume_percent": 50,
            "error": None,
            "command_id": None,
        }
        await store.set_media_status(dict(status))
        command = asyncio.create_task(store.execute_media_command("play", "media-1"))
        await asyncio.sleep(0)
        assert (await store.media())["command"] == {
            "id": "media-1",
            "action": "play",
            "issued_at": (await store.media())["command"]["issued_at"],
            "arguments": {},
            "prior_playback_state": "STOPPED",
        }

        status.update(
            {
                "playback_state": "PLAYING",
                "track_name": "Test track",
                "artist_name": "Test artist",
                "command_id": "media-1",
            }
        )
        await store.set_media_status(status)
        outcome = await command
        assert outcome.message.startswith("Playing music")
        assert outcome.state["playback_state"] == "PLAYING"
        assert (await store.media())["command"] is None
        await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("action", "prior_state", "reported_state"),
    [
        ("next", "PLAYING", "PAUSED"),
        ("next", "PAUSED", "PLAYING"),
        ("pause", "PLAYING", "STOPPED"),
    ],
)
def test_media_ack_uses_browser_state_not_stale_server_state(
    action: str, prior_state: str, reported_state: str
) -> None:
    async def scenario() -> None:
        store = WorldStore(load_config())
        status = {
            "connected": True,
            "playback_state": prior_state,
            "device_id": "browser-device",
            "track_name": "Test track",
            "artist_name": "Test artist",
            "position_ms": 1_000,
            "duration_ms": 180_000,
            "volume_percent": 50,
            "error": None,
            "command_id": None,
        }
        await store.set_media_status(dict(status))
        command = asyncio.create_task(store.execute_media_command(action, f"{action}-fresh"))
        await asyncio.sleep(0)
        status.update({"playback_state": reported_state, "command_id": f"{action}-fresh"})
        await store.set_media_status(status)
        outcome = await command
        assert outcome.state["playback_state"] == reported_state
        await store.close()

    asyncio.run(scenario())


def test_media_volume_command_carries_exact_percent_to_browser() -> None:
    async def scenario() -> None:
        store = WorldStore(load_config())
        status = {
            "connected": True,
            "playback_state": "STOPPED",
            "device_id": "browser-device",
            "track_name": None,
            "artist_name": None,
            "position_ms": 0,
            "duration_ms": 0,
            "volume_percent": 50,
            "error": None,
            "command_id": None,
        }
        await store.set_media_status(dict(status))
        command = asyncio.create_task(
            store.execute_media_command("set_volume", "volume-1", {"percent": 65})
        )
        await asyncio.sleep(0)
        assert (await store.media())["command"]["arguments"] == {"percent": 65}
        status.update({"volume_percent": 65, "command_id": "volume-1"})
        await store.set_media_status(status)
        outcome = await command
        assert outcome.message == "Spotify volume is now 65 percent."
        await store.close()

    asyncio.run(scenario())


def test_media_command_requires_a_recent_connected_browser() -> None:
    async def scenario() -> None:
        store = WorldStore(load_config())
        with pytest.raises(WayneManorApiError) as error:
            await store.execute_media_command("pause", "media-2")
        assert error.value.code == "spotify_not_connected"
        await store.close()

    asyncio.run(scenario())

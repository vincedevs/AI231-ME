from __future__ import annotations

import asyncio
import contextlib
import copy
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .config import WayneManorConfig


class WayneManorApiError(RuntimeError):
    def __init__(
        self, status_code: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class MutationOutcome:
    changed: bool
    message: str
    state: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class WorldStore:
    """Authoritative in-memory Wayne Manor state with serialized mutations."""

    def __init__(self, config: WayneManorConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._updated_at = utc_now()
        self._lights = {
            light_id: {
                "id": light_id,
                "display_name": values.display_name,
                "on": values.initial_power,
                "brightness_percent": values.initial_brightness_percent,
                "color": values.initial_color,
                "color_hex": values.supported_colors[values.initial_color],
            }
            for light_id, values in config.lights.items()
        }
        self._thermostats = {
            thermostat_id: {
                "id": thermostat_id,
                "display_name": values.display_name,
                "setpoint": values.initial_setpoint,
                "unit": values.unit,
                "minimum_setpoint": values.minimum_setpoint,
                "maximum_setpoint": values.maximum_setpoint,
            }
            for thermostat_id, values in config.thermostats.items()
        }
        self._telephone = {
            "display_name": config.telephone.display_name,
            "ringing": False,
            "call_id": None,
            "started_at": None,
            "ring_duration_seconds": config.telephone.ring_duration_seconds,
        }
        self._telephone_stop_task: asyncio.Task[None] | None = None
        self._media = {
            "connected": False,
            "playback_state": "STOPPED",
            "device_id": None,
            "track_name": None,
            "artist_name": None,
            "position_ms": 0,
            "duration_ms": 0,
            "volume_percent": config.spotify.initial_volume * 100,
            "error": None,
            "command": None,
        }
        self._media_seen_at: float | None = None
        self._media_command_waiters: dict[str, asyncio.Future[MutationOutcome]] = {}
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def _light_snapshot(self, light_id: str) -> dict[str, Any]:
        light = copy.deepcopy(self._lights[light_id])
        light["effective_intensity"] = (
            light["brightness_percent"] / 100.0 if light["on"] else 0.0
        )
        return light

    def _snapshot_unlocked(self) -> dict[str, Any]:
        return {
            "updated_at": self._updated_at,
            "default_light_group": self.config.default_light_group,
            "default_thermostat": self.config.default_thermostat,
            "lights": {
                light_id: self._light_snapshot(light_id) for light_id in self._lights
            },
            "thermostats": copy.deepcopy(self._thermostats),
            "telephone": copy.deepcopy(self._telephone),
            "media": self._media_snapshot_unlocked(),
        }

    def _media_snapshot_unlocked(self) -> dict[str, Any]:
        media = copy.deepcopy(self._media)
        if self._media_seen_at is None or (
            time.monotonic() - self._media_seen_at
            > self.config.spotify.status_timeout_seconds
        ):
            media["connected"] = False
        return media

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return self._snapshot_unlocked()

    async def light(self, light_id: str) -> dict[str, Any]:
        async with self._lock:
            self._require_light(light_id)
            return self._light_snapshot(light_id)

    async def thermostat(self, thermostat_id: str) -> dict[str, Any]:
        async with self._lock:
            self._require_thermostat(thermostat_id)
            return copy.deepcopy(self._thermostats[thermostat_id])

    async def media(self) -> dict[str, Any]:
        async with self._lock:
            return self._media_snapshot_unlocked()

    def capabilities(self) -> dict[str, Any]:
        return {
            "default_light_group": self.config.default_light_group,
            "default_thermostat": self.config.default_thermostat,
            "lights": {
                light_id: {
                    "display_name": values.display_name,
                    "supported_colors": copy.deepcopy(values.supported_colors),
                    "brightness_minimum": 0,
                    "brightness_maximum": 100,
                }
                for light_id, values in self.config.lights.items()
            },
            "thermostats": {
                thermostat_id: {
                    "display_name": values.display_name,
                    "unit": values.unit,
                    "minimum_setpoint": values.minimum_setpoint,
                    "maximum_setpoint": values.maximum_setpoint,
                    "change_sound_url": values.change_sound_url,
                    "change_sound_volume": values.change_sound_volume,
                }
                for thermostat_id, values in self.config.thermostats.items()
            },
            "telephone": {
                "display_name": self.config.telephone.display_name,
                "ring_duration_seconds": self.config.telephone.ring_duration_seconds,
                "ringtone_url": self.config.telephone.ringtone_url,
            },
            "spotify": {
                "client_id": self.config.spotify.client_id,
                "redirect_uri": self.config.spotify.redirect_uri,
                "device_name": self.config.spotify.device_name,
                "default_uri": self.config.spotify.default_uri,
                "initial_volume": self.config.spotify.initial_volume,
                "scopes": [
                    "streaming",
                    "user-read-email",
                    "user-read-private",
                    "user-modify-playback-state",
                    "playlist-read-private",
                ],
            },
        }

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def _broadcast(self, snapshot: dict[str, Any]) -> None:
        event = {"type": "state.changed", "state": snapshot}
        for queue in tuple(self._subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(copy.deepcopy(event))

    def _require_light(self, light_id: str) -> None:
        if light_id not in self._lights:
            raise WayneManorApiError(404, "light_group_not_found", f"Unknown light group: {light_id}")

    def _require_thermostat(self, thermostat_id: str) -> None:
        if thermostat_id not in self._thermostats:
            raise WayneManorApiError(
                404, "thermostat_not_found", f"Unknown thermostat: {thermostat_id}"
            )

    def _finish(self, *, changed: bool, message: str, state: dict[str, Any]) -> MutationOutcome:
        snapshot = self._snapshot_unlocked()
        outcome = MutationOutcome(changed, message, copy.deepcopy(state))
        if changed:
            self._broadcast(snapshot)
        return outcome

    def _mark_changed(self) -> None:
        self._updated_at = utc_now()

    async def set_light_power(self, light_id: str, on: bool) -> MutationOutcome:
        async with self._lock:
            self._require_light(light_id)
            light = self._lights[light_id]
            changed = light["on"] != on
            if changed:
                light["on"] = on
                self._mark_changed()
            status = "on" if on else "off"
            return self._finish(
                changed=changed,
                message=f"The {light['display_name'].lower()} lights are now {status}.",
                state=self._light_snapshot(light_id),
            )

    async def set_light_brightness(self, light_id: str, percent: float) -> MutationOutcome:
        async with self._lock:
            self._require_light(light_id)
            light = self._lights[light_id]
            changed = light["brightness_percent"] != percent
            if changed:
                light["brightness_percent"] = percent
                self._mark_changed()
            spoken = int(percent) if percent.is_integer() else percent
            return self._finish(
                changed=changed,
                message=(
                    f"The {light['display_name'].lower()} brightness is now {spoken} percent."
                ),
                state=self._light_snapshot(light_id),
            )

    async def set_light_color(self, light_id: str, color: str) -> MutationOutcome:
        async with self._lock:
            self._require_light(light_id)
            light_config = self.config.lights[light_id]
            if color not in light_config.supported_colors:
                raise WayneManorApiError(
                    422,
                    "unsupported_color",
                    f"{color.title()} is not supported by {light_config.display_name}.",
                    {"supported_colors": list(light_config.supported_colors)},
                )
            light = self._lights[light_id]
            changed = light["color"] != color
            if changed:
                light["color"] = color
                light["color_hex"] = light_config.supported_colors[color]
                self._mark_changed()
            return self._finish(
                changed=changed,
                message=f"The {light['display_name'].lower()} lights are now {color}.",
                state=self._light_snapshot(light_id),
            )

    async def set_temperature(
        self, thermostat_id: str, degrees: float, unit: str
    ) -> MutationOutcome:
        async with self._lock:
            self._require_thermostat(thermostat_id)
            thermostat = self._thermostats[thermostat_id]
            if unit != thermostat["unit"]:
                raise WayneManorApiError(
                    422,
                    "temperature_unit_mismatch",
                    f"{thermostat['display_name']} expects {thermostat['unit']}.",
                    {"expected_unit": thermostat["unit"]},
                )
            minimum = thermostat["minimum_setpoint"]
            maximum = thermostat["maximum_setpoint"]
            if not minimum <= degrees <= maximum:
                raise WayneManorApiError(
                    422,
                    "unsafe_temperature",
                    f"The temperature must be between {minimum:g} and {maximum:g} degrees.",
                    {"minimum": minimum, "maximum": maximum, "unit": unit},
                )
            changed = thermostat["setpoint"] != degrees
            if changed:
                thermostat["setpoint"] = degrees
                self._mark_changed()
            spoken = int(degrees) if degrees.is_integer() else degrees
            unit_name = "Celsius" if unit == "celsius" else "Fahrenheit"
            return self._finish(
                changed=changed,
                message=f"Wayne Manor's temperature is set to {spoken} degrees {unit_name}.",
                state=copy.deepcopy(thermostat),
            )

    async def start_telephone_call(self, call_id: str) -> MutationOutcome:
        """Ring the telephone and replace any earlier pending simulated call."""
        async with self._lock:
            if self._telephone_stop_task is not None:
                self._telephone_stop_task.cancel()
            self._telephone.update(
                {
                    "ringing": True,
                    "call_id": call_id,
                    "started_at": utc_now(),
                }
            )
            self._mark_changed()
            self._telephone_stop_task = asyncio.create_task(
                self._stop_telephone_after_delay(call_id)
            )
            return self._finish(
                changed=True,
                message=f"The {self.config.telephone.display_name.lower()} is ringing.",
                state=copy.deepcopy(self._telephone),
            )

    async def _stop_telephone_after_delay(self, call_id: str) -> None:
        try:
            await asyncio.sleep(self.config.telephone.ring_duration_seconds)
            await self.stop_telephone_call(call_id)
        except asyncio.CancelledError:
            return

    async def stop_telephone_call(self, call_id: str | None = None) -> MutationOutcome:
        """Stop the current ring; a call ID prevents an older timer stopping a newer call."""
        async with self._lock:
            if call_id is not None and self._telephone["call_id"] != call_id:
                return self._finish(
                    changed=False,
                    message="The telephone call has already changed.",
                    state=copy.deepcopy(self._telephone),
                )
            changed = bool(self._telephone["ringing"])
            if changed:
                self._telephone.update(
                    {"ringing": False, "call_id": None, "started_at": None}
                )
                self._mark_changed()
            return self._finish(
                changed=changed,
                message=f"The {self.config.telephone.display_name.lower()} has stopped ringing.",
                state=copy.deepcopy(self._telephone),
            )

    async def execute_media_command(
        self,
        action: str,
        command_id: str,
        arguments: dict[str, Any] | None = None,
    ) -> MutationOutcome:
        """Send one command to the connected browser and await its acknowledgement."""
        messages = {
            "play": "Playing music through the Wayne Manor turntable.",
            "pause": "Music paused.",
            "resume": "Music resumed.",
            "stop": "Music stopped.",
            "next": "Skipping to the next track.",
            "set_volume": "Spotify volume updated.",
        }
        if action not in messages:
            raise WayneManorApiError(404, "media_action_not_found", "Unknown media action.")
        async with self._lock:
            media = self._media_snapshot_unlocked()
            if not media["connected"]:
                raise WayneManorApiError(
                    409,
                    "spotify_not_connected",
                    "Spotify is not connected in Wayne Manor. Open it on the laptop and connect Spotify first.",
                )
            if self._media["command"] is not None:
                raise WayneManorApiError(
                    409,
                    "media_command_in_progress",
                    "Wayne Manor is still completing the previous music command.",
                )
            command = {
                "id": command_id,
                "action": action,
                "issued_at": utc_now(),
                "arguments": copy.deepcopy(arguments or {}),
                "prior_playback_state": self._media["playback_state"],
            }
            self._media["command"] = command
            waiter: asyncio.Future[MutationOutcome] = (
                asyncio.get_running_loop().create_future()
            )
            self._media_command_waiters[command_id] = waiter
            self._mark_changed()
            self._finish(changed=True, message=messages[action], state=self._media)

        try:
            return await asyncio.wait_for(
                asyncio.shield(waiter),
                timeout=self.config.spotify.command_timeout_seconds,
            )
        except TimeoutError as error:
            async with self._lock:
                self._media_command_waiters.pop(command_id, None)
                if not waiter.done():
                    waiter.cancel()
                current_command = self._media.get("command")
                if isinstance(current_command, dict) and current_command.get("id") == command_id:
                    self._media["command"] = None
                    self._media["error"] = "The laptop did not acknowledge the music command."
                    self._mark_changed()
                    self._finish(
                        changed=True,
                        message=self._media["error"],
                        state=self._media,
                    )
            raise WayneManorApiError(
                504,
                "media_command_timeout",
                "The Wayne Manor Spotify player did not respond in time.",
            ) from error

    async def set_media_status(self, status: dict[str, Any]) -> MutationOutcome:
        """Record browser playback state and resolve a matching pending command."""
        command_id = status.pop("command_id", None)
        error = status.get("error")
        async with self._lock:
            self._media_seen_at = time.monotonic()
            command = self._media.get("command")
            acknowledged = (
                command_id is not None
                and isinstance(command, dict)
                and command.get("id") == command_id
            )
            comparable = {key: value for key, value in self._media.items() if key != "command"}
            changed = comparable != status or acknowledged
            self._media.update(status)
            if acknowledged:
                self._media["command"] = None
            if self._media["playback_state"] == "STOPPED":
                self._media.update(
                    {
                        "track_name": None,
                        "artist_name": None,
                        "position_ms": 0,
                        "duration_ms": 0,
                    }
                )
            if changed:
                self._mark_changed()
            message = error or "Spotify playback status updated."
            outcome = self._finish(
                changed=changed,
                message=message,
                state=self._media_snapshot_unlocked(),
            )
            if acknowledged:
                waiter = self._media_command_waiters.pop(command_id, None)
                if waiter is not None and not waiter.done():
                    if error:
                        waiter.set_exception(
                            WayneManorApiError(502, "spotify_command_failed", str(error))
                        )
                    else:
                        action = command["action"]
                        accepted_states = {
                            "play": {"PLAYING"},
                            "resume": {"PLAYING"},
                            # The browser player's fresh state is authoritative;
                            # the API snapshot can lag behind external Spotify
                            # controls or a recent SDK event.
                            "pause": {"PAUSED", "STOPPED"},
                            "stop": {"STOPPED"},
                            # Skip preserves whichever state the browser
                            # actually had, not the server's possibly stale one.
                            "next": {"PLAYING", "PAUSED"},
                        }.get(action)
                        actual_state = outcome.state.get("playback_state")
                        if accepted_states is not None and actual_state not in accepted_states:
                            waiter.set_exception(
                                WayneManorApiError(
                                    502,
                                    "spotify_state_mismatch",
                                    "Spotify did not reach the requested playback state.",
                                )
                            )
                            return outcome
                        success_messages = {
                            "play": "Playing music through the Wayne Manor turntable.",
                            "pause": "Music paused.",
                            "resume": "Music resumed.",
                            "stop": "Music stopped.",
                            "next": "Skipping to the next track.",
                            "set_volume": (
                                f"Spotify volume is now "
                                f"{outcome.state['volume_percent']:g} percent."
                            ),
                        }
                        waiter.set_result(
                            MutationOutcome(
                                changed=True,
                                message=success_messages[action],
                                state=copy.deepcopy(outcome.state),
                            )
                        )
            return outcome

    async def close(self) -> None:
        task = self._telephone_stop_task
        self._telephone_stop_task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for waiter in self._media_command_waiters.values():
            if not waiter.done():
                waiter.cancel()
        self._media_command_waiters.clear()

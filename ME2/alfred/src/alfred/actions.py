from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any

from .config import AlfredConfig
from .contracts import ActionRequest, ActionResult
from .google_chat import GoogleChatClient, GoogleChatError, MockGoogleChatClient
from .linphone_call import LinphoneCallController, LinphoneCallError, MockCallController
from .reminders import ReminderError, ReminderStore
from .scheduling import MockAlertScheduler, SchedulingError, SystemdAlertScheduler
from .spotify import (
    MockSpotifyController,
    SpotifyError,
    WayneManorSpotifyController,
)
from .system_actions import SystemActionError, VolumeController, spoken_local_time
from .tts import prepare_tts_text
from .wayne_manor import WayneManorClient, WayneManorError
from .weather import OpenMeteoClient, WeatherError

LOGGER = logging.getLogger(__name__)


def _surface(request: ActionRequest, slot: str) -> str:
    value = request.slots[slot]
    return str(value.get("surface", "")).strip() if isinstance(value, dict) else str(value)


def _mock_handlers() -> dict[str, Callable[[ActionRequest], str]]:
    return {
        "start_call": lambda request: "Mock action completed. A call would start.",
        "set_timer": lambda request: (
            f"Mock action completed. The timer value would be {_surface(request, 'duration')}."
        ),
        "set_alarm": lambda request: (
            f"Mock action completed. The alarm value would be {_surface(request, 'time')}."
        ),
    }


def _mock_wayne_manor_handlers() -> dict[str, Callable[[ActionRequest], str]]:
    """Keep isolated state-machine tests independent of a running Wayne Manor."""
    return {
        "turn_on_wayne_manor_lights": lambda request: (
            "Mock action completed. The lights would turn on."
        ),
        "turn_off_wayne_manor_lights": lambda request: (
            "Mock action completed. The lights would turn off."
        ),
        "set_wayne_manor_temperature": lambda request: (
            f"Wayne Manor's temperature is set to {_surface(request, 'degrees')}."
        ),
        "set_wayne_manor_brightness": lambda request: (
            f"The living room brightness is now {_surface(request, 'percent')}."
        ),
        "set_wayne_manor_color": lambda request: (
            f"The living room lights are now {_surface(request, 'color')}."
        ),
    }


def _mock_message_handlers() -> dict[str, Callable[[ActionRequest], str]]:
    """Keep isolated state-machine tests from contacting Google Chat."""
    return {
        "send_message": lambda request: (
            "Mock action completed. A Google Chat message would be sent."
        )
    }


class MockActionExecutor:
    """Deterministic handlers used by isolated application tests."""

    def __init__(self) -> None:
        self.history: list[ActionRequest] = []
        self.media_pause_requests = 0
        self.spotify = MockSpotifyController()
        self.handlers = {
            **_mock_handlers(),
            **_mock_wayne_manor_handlers(),
            **_mock_message_handlers(),
            "play_music": lambda request: self.spotify.play(),
            "pause_music": lambda request: self.spotify.pause(),
            "stop_music": lambda request: self.spotify.stop(),
            "next_music": lambda request: self.spotify.next(),
        }

    def pause_media_for_interaction(self) -> None:
        self.media_pause_requests += 1
        self.spotify.pause_for_interaction()

    def resume_media_after_interaction(self) -> None:
        self.spotify.resume_after_interaction()

    def execute(self, request: ActionRequest) -> ActionResult:
        handler = self.handlers.get(request.handler)
        if handler is None:
            raise ValueError(f"No approved mock handler exists for {request.handler!r}")
        started = time.perf_counter()
        message = handler(request)
        self.history.append(request)
        return ActionResult(
            request_id=request.request_id,
            status="success",
            message=message,
            handler=request.handler,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


class ActionExecutor:
    """Execute the 19 reviewed intent handlers through explicit local backends."""

    def __init__(
        self,
        *,
        weather: OpenMeteoClient,
        volume: VolumeController,
        reminders: ReminderStore,
        wayne_manor: WayneManorClient,
        call_backend: LinphoneCallController | WayneManorClient | MockCallController,
        message_backend: GoogleChatClient | MockGoogleChatClient,
        spotify: WayneManorSpotifyController | MockSpotifyController,
        scheduler: SystemdAlertScheduler | MockAlertScheduler,
        fallback_colors: tuple[str, ...],
        fallback_color_seed: int,
        fallback_messages: tuple[str, ...],
        fallback_message_seed: int | None,
        reminder_spoken_limit: int | None,
        clock: Callable[[], Any] | None = None,
    ) -> None:
        if reminder_spoken_limit is not None and reminder_spoken_limit <= 0:
            raise ValueError("Reminder spoken limit must be positive")
        if not fallback_colors:
            raise ValueError("At least one fallback color must be configured")
        if not 1 <= len(fallback_messages) <= 5:
            raise ValueError("Configure between one and five fallback messages")
        if any(not message.strip() for message in fallback_messages):
            raise ValueError("Fallback messages cannot be empty")
        if len(set(fallback_messages)) != len(fallback_messages):
            raise ValueError("Fallback messages must be unique")
        self.weather = weather
        self.volume = volume
        self.reminders = reminders
        self.wayne_manor = wayne_manor
        self.call_backend = call_backend
        self.message_backend = message_backend
        self.spotify = spotify
        self.scheduler = scheduler
        self.fallback_colors = fallback_colors
        self.color_random = random.Random(fallback_color_seed)
        self.last_fallback_color: str | None = None
        self.fallback_messages = fallback_messages
        self.message_random = random.Random(fallback_message_seed)
        self.last_fallback_message: str | None = None
        self.reminder_spoken_limit = reminder_spoken_limit
        self.clock = clock
        # Populated once by the startup API sweep. True means commands must
        # fail immediately without attempting another network request.
        self.api_unavailable = {
            "wayne_manor": False,
            "open_meteo": False,
            "google_chat": False,
        }
        self.history: list[ActionRequest] = []
        self.media_pause_requests = 0
        self.handlers: dict[str, Callable[[ActionRequest], str]] = {
            **_mock_handlers(),
            "start_call": self._start_call,
            "set_timer": self._set_timer,
            "set_alarm": self._set_alarm,
            "play_music": lambda request: self._spotify_command("play"),
            "pause_music": lambda request: self._spotify_command("pause"),
            "stop_music": lambda request: self._spotify_command("stop"),
            "next_music": lambda request: self._spotify_command("next"),
            "get_weather": self._weather,
            "get_time": self._time,
            "increase_volume": lambda request: self._volume(1),
            "decrease_volume": lambda request: self._volume(-1),
            "send_message": self._send_message,
            "create_reminder": self._create_reminder,
            "list_reminders": self._list_reminders,
            "turn_on_wayne_manor_lights": lambda request: self._wayne_manor_power(request, True),
            "turn_off_wayne_manor_lights": lambda request: self._wayne_manor_power(request, False),
            "set_wayne_manor_temperature": self._wayne_manor_temperature,
            "set_wayne_manor_brightness": self._wayne_manor_brightness,
            "set_wayne_manor_color": self._wayne_manor_color,
        }

    def check_api_availability(self, timeout_seconds: float) -> dict[str, bool]:
        """Run the one-time, non-mutating startup API sweep."""
        if timeout_seconds <= 0:
            raise ValueError("API probe timeout must be positive")
        available = {
            "wayne_manor": self.wayne_manor.check_availability(timeout_seconds),
            "open_meteo": self.weather.check_availability(timeout_seconds),
            "google_chat": (
                True
                if isinstance(self.message_backend, MockGoogleChatClient)
                else self.message_backend.check_availability(timeout_seconds)
            ),
        }
        self.api_unavailable = {name: not status for name, status in available.items()}
        LOGGER.info(
            "api_startup_sweep wayne_manor=%s open_meteo=%s google_chat=%s",
            "available" if available["wayne_manor"] else "unavailable",
            "available" if available["open_meteo"] else "unavailable",
            "available" if available["google_chat"] else "unavailable",
        )
        return available

    def _require_weather(self) -> None:
        if self.api_unavailable["open_meteo"]:
            raise WeatherError("The weather service is unavailable right now.")

    def _require_google_chat(self) -> None:
        if self.api_unavailable["google_chat"]:
            raise GoogleChatError("Google Chat is unavailable right now.")

    def _require_wayne_manor(self) -> None:
        if self.api_unavailable["wayne_manor"]:
            raise WayneManorError("Wayne Manor is unavailable right now.")

    def _start_call(self, request: ActionRequest) -> str:
        if isinstance(self.call_backend, WayneManorClient):
            self._require_wayne_manor()
        return self.call_backend.call(request.request_id)

    def _spotify_command(self, action: str) -> str:
        if isinstance(self.spotify, WayneManorSpotifyController):
            self._require_wayne_manor()
        commands = {
            "play": self.spotify.play,
            "pause": self.spotify.pause,
            "stop": self.spotify.stop,
            "next": self.spotify.next,
        }
        return commands[action]()

    def pause_media_for_interaction(self) -> None:
        self.media_pause_requests += 1
        if (
            isinstance(self.spotify, WayneManorSpotifyController)
            and self.api_unavailable["wayne_manor"]
        ):
            LOGGER.info("spotify_interaction_pause_skipped reason=wayne_manor_startup_unavailable")
            return
        try:
            self.spotify.pause_for_interaction()
        except SpotifyError as error:
            LOGGER.info("spotify_interaction_pause_skipped reason=%s", error)

    def resume_media_after_interaction(self) -> None:
        if (
            isinstance(self.spotify, WayneManorSpotifyController)
            and self.api_unavailable["wayne_manor"]
        ):
            return
        try:
            self.spotify.resume_after_interaction()
        except SpotifyError as error:
            LOGGER.warning("spotify_interaction_resume_failed reason=%s", error)

    def _weather(self, request: ActionRequest) -> str:
        self._require_weather()
        location_query = request.payload.get("location_query")
        return self.weather.spoken_summary_for(
            location_query if isinstance(location_query, str) else None
        )

    def _time(self, request: ActionRequest) -> str:
        del request
        return spoken_local_time(None if self.clock is None else self.clock())

    def _volume(self, direction: int) -> str:
        if isinstance(self.spotify, WayneManorSpotifyController):
            self._require_wayne_manor()
            return self.spotify.adjust_volume(direction, self.volume.step_percent)
        level = self.volume.adjust(direction)
        verb = "increased" if direction > 0 else "decreased"
        return f"Volume {verb} to {level} percent."

    def _create_reminder(self, request: ActionRequest) -> str:
        reminder = self.reminders.create(request.slots["task"])
        return f"I've added your reminder to {reminder.surface_text}."

    def _set_timer(self, request: ActionRequest) -> str:
        slot = request.slots["duration"]
        return self.scheduler.set_timer(
            int(slot["seconds"]), str(slot["surface"]), request.request_id
        )

    def _set_alarm(self, request: ActionRequest) -> str:
        return self.scheduler.set_alarm(str(request.slots["time"]["text"]), request.request_id)

    def _send_message(self, request: ActionRequest) -> str:
        if isinstance(self.message_backend, GoogleChatClient):
            self._require_google_chat()
        recipient = request.payload.get("recipient")
        message = request.payload.get("message")
        if message is None:
            available = [
                item for item in self.fallback_messages if item != self.last_fallback_message
            ]
            message = self.message_random.choice(available or list(self.fallback_messages))
            self.last_fallback_message = message
            LOGGER.info("message_fallback_selected")
        if recipient is not None and not isinstance(recipient, str):
            raise GoogleChatError("I couldn't prepare your message, so nothing was sent.")
        if not isinstance(message, str):
            raise GoogleChatError("I couldn't prepare your message, so nothing was sent.")
        return self.message_backend.send_message(
            message,
            request.request_id,
            recipient=recipient,
        )

    def _list_reminders(self, request: ActionRequest) -> str:
        del request
        reminders, total = self.reminders.list_active(self.reminder_spoken_limit)
        if total == 0:
            return "You have no active reminders."
        noun = "reminder" if total == 1 else "reminders"
        if total == 1:
            item_lines = [f"{prepare_tts_text(reminders[0].surface_text)}."]
        else:
            item_lines = [
                f"Reminder {index + 1}. {prepare_tts_text(item.surface_text)}."
                for index, item in enumerate(reminders)
            ]
        remaining = total - len(reminders)
        lines = [f"You have {total} {noun}.", *item_lines]
        if remaining:
            lines.append(f"There are {remaining} more reminder{'s' if remaining != 1 else ''}.")
        return "\n".join(lines)

    def _wayne_manor_power(self, request: ActionRequest, on: bool) -> str:
        self._require_wayne_manor()
        return self.wayne_manor.set_light_power(on, request.request_id)

    def _wayne_manor_temperature(self, request: ActionRequest) -> str:
        self._require_wayne_manor()
        degrees = float(request.slots["degrees"]["value"])
        return self.wayne_manor.set_temperature(degrees, request.request_id)

    def _wayne_manor_brightness(self, request: ActionRequest) -> str:
        self._require_wayne_manor()
        percent = float(request.slots["percent"]["value"])
        return self.wayne_manor.set_brightness(percent, request.request_id)

    def _wayne_manor_color(self, request: ActionRequest) -> str:
        self._require_wayne_manor()
        slot = request.slots["color"]
        color = str(slot["text"])
        if slot.get("source") == "fallback_default":
            available = [item for item in self.fallback_colors if item != self.last_fallback_color]
            color = self.color_random.choice(available or list(self.fallback_colors))
            self.last_fallback_color = color
        return self.wayne_manor.set_color(color, request.request_id)

    def execute(self, request: ActionRequest) -> ActionResult:
        handler = self.handlers.get(request.handler)
        if handler is None:
            raise ValueError(f"No approved action handler exists for {request.handler!r}")
        started = time.perf_counter()
        try:
            message = handler(request)
            status = "success"
        except (
            LinphoneCallError,
            GoogleChatError,
            WayneManorError,
            ReminderError,
            SchedulingError,
            SpotifyError,
            SystemActionError,
            WeatherError,
        ) as error:
            message = str(error)
            status = "failure"
        self.history.append(request)
        return ActionResult(
            request_id=request.request_id,
            status=status,
            message=message,
            handler=request.handler,
            duration_ms=(time.perf_counter() - started) * 1000,
        )


def build_action_executor(
    config: AlfredConfig, *, mock_device_actions: bool = False
) -> ActionExecutor:
    """Build real backends by default or local test doubles when requested."""
    actions = config.document["actions"]
    weather_values = actions["weather"]
    volume_values = actions["volume"]
    reminder_values = actions["reminders"]
    wayne_manor_values = actions["wayne_manor"]
    call_values = actions["call"]
    message_values = actions["google_chat"]
    scheduler_values = actions["scheduler"]
    weather = OpenMeteoClient(
        location_name=str(weather_values["location_name"]),
        latitude=float(weather_values["latitude"]),
        longitude=float(weather_values["longitude"]),
        temperature_unit=str(weather_values["temperature_unit"]),
        timeout_seconds=float(weather_values["timeout_seconds"]),
        cache_seconds=float(weather_values["cache_seconds"]),
        maximum_location_characters=int(weather_values["maximum_location_characters"]),
    )
    volume = VolumeController(
        step_percent=int(volume_values["step_percent"]),
        linux_maximum_percent=int(volume_values["linux_maximum_percent"]),
        timeout_seconds=float(volume_values["timeout_seconds"]),
    )
    reminders = ReminderStore(
        config.local_path(str(actions["database_path"])),
        max_task_characters=int(reminder_values["max_task_characters"]),
    )
    wayne_manor_url_environment = str(wayne_manor_values["base_url_environment"])
    wayne_manor = WayneManorClient(
        base_url=os.environ.get(
            wayne_manor_url_environment,
            str(wayne_manor_values["base_url"]),
        ),
        light_group=str(wayne_manor_values["default_light_group"]),
        thermostat=str(wayne_manor_values["default_thermostat"]),
        temperature_unit=str(wayne_manor_values["temperature_unit"]),
        timeout_seconds=float(wayne_manor_values["timeout_seconds"]),
    )
    call_backend_name = str(call_values["backend"])
    if mock_device_actions:
        call_backend = MockCallController()
    elif call_backend_name == "wayne_manor":
        call_backend = wayne_manor
    else:
        destination_environment = str(call_values["linphone_destination_environment"])
        destination = os.environ.get(destination_environment, "")
        call_backend = LinphoneCallController(
            destination=destination,
            executable=str(call_values["linphone_executable"]),
            timeout_seconds=float(call_values["command_timeout_seconds"]),
        )
    message_backend = (
        MockGoogleChatClient()
        if mock_device_actions
        else GoogleChatClient(
            webhook_url=os.environ.get(str(message_values["webhook_url_environment"])),
            maximum_recipient_characters=int(message_values["maximum_recipient_characters"]),
            maximum_message_characters=int(message_values["maximum_message_characters"]),
            timeout_seconds=float(message_values["timeout_seconds"]),
        )
    )
    if mock_device_actions:
        spotify = MockSpotifyController()
    else:
        spotify = WayneManorSpotifyController(wayne_manor)
    scheduler = (
        MockAlertScheduler()
        if mock_device_actions
        else SystemdAlertScheduler(
            maximum_timer_seconds=int(scheduler_values["maximum_timer_seconds"]),
            command_timeout_seconds=float(scheduler_values["command_timeout_seconds"]),
        )
    )
    return ActionExecutor(
        weather=weather,
        volume=volume,
        reminders=reminders,
        wayne_manor=wayne_manor,
        call_backend=call_backend,
        message_backend=message_backend,
        spotify=spotify,
        scheduler=scheduler,
        fallback_colors=tuple(str(item) for item in wayne_manor_values["fallback_colors"]),
        fallback_color_seed=int(wayne_manor_values["fallback_color_seed"]),
        fallback_messages=tuple(str(item) for item in message_values["fallback_messages"]),
        fallback_message_seed=(
            None
            if message_values["fallback_message_seed"] is None
            else int(message_values["fallback_message_seed"])
        ),
        reminder_spoken_limit=(
            None
            if reminder_values["spoken_limit"] is None
            else int(reminder_values["spoken_limit"])
        ),
    )

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise

import pytest

from alfred.actions import ActionExecutor, MockActionExecutor, build_action_executor
from alfred.config import load_config
from alfred.contracts import ActionRequest, Decision, VcmResult
from alfred.google_chat import MockGoogleChatClient
from alfred.linphone_call import MockCallController
from alfred.policy import apply_policy, normalize_slot
from alfred.reminders import ReminderStore
from alfred.scheduling import MockAlertScheduler
from alfred.spotify import MockSpotifyController, WayneManorSpotifyController

SLOTS = {
    "TIMER": {"duration": {"surface": "ten minutes"}},
    "ALARM": {"time": {"surface": "seven am"}},
    "TEMPERATURE": {"degrees": {"surface": "22 degrees"}},
    "BRIGHTNESS": {"percent": {"surface": "60 percent"}},
    "COLOR": {"color": {"surface": "blue"}},
    "CREATE_REMINDER": {"task": {"surface": "drink water"}},
}

FALLBACK_MESSAGES = (
    "Fallback message one.",
    "Fallback message two.",
    "Fallback message three.",
)


def test_wayne_manor_url_can_be_overridden_from_private_environment(monkeypatch) -> None:
    monkeypatch.setenv("ALFRED_WAYNE_MANOR_URL", "http://192.0.2.10:8765/api/v1")
    executor = build_action_executor(load_config(), mock_device_actions=True)
    assert executor.wayne_manor.base_url == "http://192.0.2.10:8765/api/v1"


def test_local_development_mode_mocks_chargeable_and_device_backends() -> None:
    executor = build_action_executor(load_config(), mock_device_actions=True)
    assert isinstance(executor.spotify, MockSpotifyController)
    assert isinstance(executor.scheduler, MockAlertScheduler)
    assert isinstance(executor.message_backend, MockGoogleChatClient)
    assert isinstance(executor.call_backend, MockCallController)


def test_default_music_and_volume_backend_is_wayne_manor() -> None:
    executor = build_action_executor(load_config())
    assert isinstance(executor.spotify, WayneManorSpotifyController)


def test_startup_api_sweep_sets_flags_and_network_actions_fail_fast(monkeypatch) -> None:
    executor = build_action_executor(load_config())
    monkeypatch.setattr(executor.wayne_manor, "check_availability", lambda timeout: False)
    monkeypatch.setattr(executor.weather, "check_availability", lambda timeout: False)
    monkeypatch.setattr(executor.message_backend, "check_availability", lambda timeout: False)

    assert executor.check_api_availability(0.25) == {
        "wayne_manor": False,
        "open_meteo": False,
        "google_chat": False,
    }
    assert executor.api_unavailable == {
        "wayne_manor": True,
        "open_meteo": True,
        "google_chat": True,
    }

    # Wake-time Spotify handling must not touch the unreachable API.
    monkeypatch.setattr(
        executor.spotify,
        "pause_for_interaction",
        lambda: pytest.fail("unavailable Wayne Manor must not be called"),
    )
    executor.pause_media_for_interaction()

    assert executor.execute(request("get_weather")).message == (
        "The weather service is unavailable right now."
    )
    assert executor.execute(request("send_message")).message == (
        "Google Chat is unavailable right now."
    )
    assert executor.execute(request("play_music")).message == (
        "Wayne Manor is unavailable right now."
    )
    assert executor.execute(request("turn_on_wayne_manor_lights")).message == (
        "Wayne Manor is unavailable right now."
    )


@pytest.mark.parametrize("intent", load_config().document["actions"]["enabled_intents"])
def test_every_intent_reaches_exactly_one_reviewed_handler(intent: str) -> None:
    config = load_config()
    result = VcmResult(Decision.EXECUTE, intent, 0.99, SLOTS.get(intent, {}), 0.99)
    outcome = apply_policy(result, config)
    assert outcome.decision is Decision.EXECUTE
    assert outcome.action_request is not None
    assert outcome.action_request.handler == config.action_by_intent[intent]["handler"]


@pytest.mark.parametrize(
    "intent",
    [
        "PLAY_MUSIC",
        "PAUSE",
        "STOP",
        "NEXT",
        "CALL",
    ],
)
def test_media_and_call_handlers_have_deterministic_test_doubles(intent: str) -> None:
    config = load_config()
    outcome = apply_policy(
        VcmResult(Decision.EXECUTE, intent, 0.99, SLOTS.get(intent, {}), 0.99), config
    )
    assert outcome.action_request is not None
    executor = MockActionExecutor()
    assert executor.execute(outcome.action_request).status == "success"


@pytest.mark.parametrize(
    "intent",
    ["TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR"],
)
def test_slot_dependent_intents_execute_with_valid_slots(intent: str) -> None:
    config = load_config()
    outcome = apply_policy(VcmResult(Decision.EXECUTE, intent, 0.99, SLOTS[intent], 0.99), config)
    assert outcome.decision is Decision.EXECUTE
    assert outcome.action_request is not None


@pytest.mark.parametrize("intent", ["TEMPERATURE", "BRIGHTNESS", "COLOR"])
def test_wayne_manor_intents_never_silently_substitute_a_slot(intent) -> None:
    config = load_config()
    outcome = apply_policy(VcmResult(Decision.EXECUTE, intent, 0.99, {}, 0.99), config)
    assert outcome.decision is Decision.LOW_CONFIDENCE
    assert outcome.action_request is None


def test_invalid_slots_never_reach_an_action() -> None:
    config = load_config()
    unsafe_temperature = apply_policy(
        VcmResult(
            Decision.EXECUTE,
            "TEMPERATURE",
            0.99,
            {"degrees": {"surface": "99 degrees"}},
            0.99,
        ),
        config,
    )
    unsupported_color = apply_policy(
        VcmResult(
            Decision.EXECUTE,
            "COLOR",
            0.99,
            {"color": {"surface": "infrared"}},
            0.99,
        ),
        config,
    )
    assert unsafe_temperature.decision is Decision.LOW_CONFIDENCE
    assert unsafe_temperature.action_request is None
    assert unsupported_color.decision is Decision.LOW_CONFIDENCE
    assert unsupported_color.action_request is None
    for intent in ("TIMER", "ALARM"):
        outcome = apply_policy(VcmResult(Decision.EXECUTE, intent, 0.99, {}, 0.99), config)
        assert outcome.decision is Decision.LOW_CONFIDENCE
        assert outcome.action_request is None


def test_create_reminder_requires_a_valid_task_even_with_fallback_enabled() -> None:
    config = load_config()
    valid = apply_policy(
        VcmResult(
            Decision.EXECUTE,
            "CREATE_REMINDER",
            0.99,
            {"task": {"surface": "submit thesis", "source": "clarification_asr"}},
            0.99,
        ),
        config,
    )
    missing = apply_policy(VcmResult(Decision.EXECUTE, "CREATE_REMINDER", 0.99, {}, 0.99), config)
    assert valid.decision is Decision.EXECUTE
    assert valid.action_request is not None
    assert valid.action_request.slots["task"]["source"] == "clarification_asr"
    assert missing.decision is Decision.LOW_CONFIDENCE


class FakeWeather:
    def spoken_summary_for(self, location_query: str | None) -> str:
        return f"{location_query or 'Quezon City'} weather."


class FakeVolume:
    def __init__(self) -> None:
        self.level = 50

    def adjust(self, direction: int) -> int:
        self.level += direction * 5
        return self.level


class FakeWayneManor:
    def set_light_power(self, on: bool, request_id: str) -> str:
        del request_id
        return f"Lights {'on' if on else 'off'}."

    def set_brightness(self, percent: float, request_id: str) -> str:
        del request_id
        return f"Brightness {percent:g}."

    def set_color(self, color: str, request_id: str) -> str:
        del request_id
        return f"Color {color}."

    def set_temperature(self, degrees: float, request_id: str) -> str:
        del request_id
        return f"Temperature {degrees:g}."


class FakeCommunications:
    def __init__(self) -> None:
        self.sent: list[tuple[str | None, str, str]] = []

    def call(self, request_id: str) -> str:
        del request_id
        return "The call has started."

    def send_message(self, message: str, request_id: str, *, recipient: str | None = None) -> str:
        self.sent.append((recipient, message, request_id))
        if recipient is not None:
            return f"Your message to {recipient} has been sent."
        return "Your message has been sent."


def request(
    handler: str,
    slots: dict | None = None,
    payload: dict | None = None,
) -> ActionRequest:
    return ActionRequest.create(
        schema_version="1.0",
        intent="TEST",
        handler=handler,
        slots=slots or {},
        payload=payload,
        intent_confidence=0.99,
        in_scope_score=0.99,
    )


def test_real_handlers_return_spoken_results_and_persist_reminders(tmp_path) -> None:
    communications = FakeCommunications()
    executor = ActionExecutor(
        weather=FakeWeather(),
        volume=FakeVolume(),
        reminders=ReminderStore(tmp_path / "state.sqlite3"),
        wayne_manor=FakeWayneManor(),
        call_backend=communications,
        message_backend=communications,
        spotify=MockSpotifyController(),
        scheduler=MockAlertScheduler(),
        fallback_colors=("red", "green", "blue"),
        fallback_color_seed=231,
        fallback_messages=FALLBACK_MESSAGES,
        fallback_message_seed=231,
        reminder_spoken_limit=5,
        clock=lambda: datetime(2026, 9, 25, 14, 3, tzinfo=UTC),
    )
    assert executor.execute(request("get_weather")).message == "Quezon City weather."
    assert (
        executor.execute(request("get_weather", payload={"location_query": "Makati City"})).message
        == "Makati City weather."
    )
    assert executor.execute(request("get_time")).message == "It is 2:03 PM."
    assert (
        executor.execute(request("play_music")).message
        == "Mock action completed. Music would start."
    )
    assert executor.execute(request("pause_music")).message == (
        "Mock action completed. Playback would pause."
    )
    assert executor.execute(request("stop_music")).message == (
        "Mock action completed. Playback would stop."
    )
    assert executor.execute(request("next_music")).message == (
        "Mock action completed. Playback would advance."
    )
    assert executor.execute(request("increase_volume")).message == "Volume increased to 55 percent."
    assert executor.execute(request("decrease_volume")).message == "Volume decreased to 50 percent."
    assert executor.execute(request("start_call")).message == "The call has started."
    assert (
        executor.execute(
            request("set_timer", {"duration": {"surface": "ten minutes", "seconds": 600}})
        ).message
        == "Mock action completed. A timer would be set for ten minutes."
    )
    assert (
        executor.execute(
            request("set_alarm", {"time": {"surface": "seven am", "text": "seven am"}})
        ).message
        == "Mock action completed. An alarm would be set for seven am."
    )
    message_request = request(
        "send_message",
        payload={"recipient": "Dad", "message": "I will be home at six."},
    )
    sent = executor.execute(message_request)
    assert sent.message == "Your message to Dad has been sent."
    assert communications.sent == [("Dad", "I will be home at six.", message_request.request_id)]
    created = executor.execute(
        request("create_reminder", {"task": {"surface": "Submit thesis", "text": "submit thesis"}})
    )
    assert created.message == "I've added your reminder to Submit thesis."
    listed = executor.execute(request("list_reminders"))
    assert listed.message == "You have 1 reminder.\nSubmit thesis."
    assert executor.execute(request("turn_on_wayne_manor_lights")).message == "Lights on."
    assert executor.execute(request("turn_off_wayne_manor_lights")).message == "Lights off."
    assert (
        executor.execute(
            request("set_wayne_manor_brightness", {"percent": {"value": 60.0}})
        ).message
        == "Brightness 60."
    )
    assert (
        executor.execute(request("set_wayne_manor_color", {"color": {"text": "blue"}})).message
        == "Color blue."
    )
    assert (
        executor.execute(
            request("set_wayne_manor_temperature", {"degrees": {"value": 22.0}})
        ).message
        == "Temperature 22."
    )


def test_message_action_without_follow_up_payload_uses_configured_fallback(tmp_path) -> None:
    communications = FakeCommunications()
    executor = ActionExecutor(
        weather=FakeWeather(),
        volume=FakeVolume(),
        reminders=ReminderStore(tmp_path / "state.sqlite3"),
        wayne_manor=FakeWayneManor(),
        call_backend=communications,
        message_backend=communications,
        spotify=MockSpotifyController(),
        scheduler=MockAlertScheduler(),
        fallback_colors=("red", "green", "blue"),
        fallback_color_seed=231,
        fallback_messages=FALLBACK_MESSAGES,
        fallback_message_seed=231,
        reminder_spoken_limit=5,
    )

    result = executor.execute(request("send_message"))

    assert result.status == "success"
    assert result.message == "Your message has been sent."
    assert len(communications.sent) == 1
    assert communications.sent[0][0] is None
    assert communications.sent[0][1] in FALLBACK_MESSAGES


def test_fallback_messages_are_randomized_without_immediate_repetition(tmp_path) -> None:
    communications = FakeCommunications()
    executor = ActionExecutor(
        weather=FakeWeather(),
        volume=FakeVolume(),
        reminders=ReminderStore(tmp_path / "state.sqlite3"),
        wayne_manor=FakeWayneManor(),
        call_backend=communications,
        message_backend=communications,
        spotify=MockSpotifyController(),
        scheduler=MockAlertScheduler(),
        fallback_colors=("red", "green", "blue"),
        fallback_color_seed=231,
        fallback_messages=FALLBACK_MESSAGES,
        fallback_message_seed=231,
        reminder_spoken_limit=5,
    )

    for _ in range(8):
        executor.execute(request("send_message"))

    messages = [message for _, message, _ in communications.sent]
    assert set(messages) <= set(FALLBACK_MESSAGES)
    assert all(left != right for left, right in pairwise(messages))


def test_fallback_color_is_pseudorandom_and_never_immediately_repeats(tmp_path) -> None:
    executor = ActionExecutor(
        weather=FakeWeather(),
        volume=FakeVolume(),
        reminders=ReminderStore(tmp_path / "state.sqlite3"),
        wayne_manor=FakeWayneManor(),
        call_backend=FakeCommunications(),
        message_backend=FakeCommunications(),
        spotify=MockSpotifyController(),
        scheduler=MockAlertScheduler(),
        fallback_colors=("red", "green", "blue"),
        fallback_color_seed=231,
        fallback_messages=FALLBACK_MESSAGES,
        fallback_message_seed=231,
        reminder_spoken_limit=5,
    )
    fallback = {
        "color": {"surface": "a surprise color", "text": "automatic", "source": "fallback_default"}
    }
    colors = [
        executor.execute(request("set_wayne_manor_color", fallback)).message for _ in range(6)
    ]
    assert all(left != right for left, right in pairwise(colors))


def test_action_executor_rejects_an_empty_fallback_color_palette(tmp_path) -> None:
    with pytest.raises(ValueError, match="fallback color"):
        ActionExecutor(
            weather=FakeWeather(),
            volume=FakeVolume(),
            reminders=ReminderStore(tmp_path / "state.sqlite3"),
            wayne_manor=FakeWayneManor(),
            call_backend=FakeCommunications(),
            message_backend=FakeCommunications(),
            spotify=MockSpotifyController(),
            scheduler=MockAlertScheduler(),
            fallback_colors=(),
            fallback_color_seed=231,
            fallback_messages=FALLBACK_MESSAGES,
            fallback_message_seed=231,
            reminder_spoken_limit=5,
        )


def test_rejections_cannot_construct_an_action() -> None:
    config = load_config()
    unsupported = apply_policy(VcmResult(Decision.UNSUPPORTED, None, 0.2, {}, 0.1), config)
    low = apply_policy(VcmResult(Decision.EXECUTE, "LIGHT_ON", 0.2, {}, 0.9), config)
    missing_slot = apply_policy(VcmResult(Decision.EXECUTE, "TIMER", 0.99, {}, 0.99), config)
    assert unsupported.action_request is None
    assert low.action_request is None
    assert missing_slot.action_request is None
    assert unsupported.decision is Decision.UNSUPPORTED
    assert low.decision is Decision.LOW_CONFIDENCE
    assert missing_slot.decision is Decision.LOW_CONFIDENCE


def test_slot_normalization_rejects_unsafe_values() -> None:
    assert normalize_slot("duration", "two minutes") == {
        "surface": "two minutes",
        "seconds": 120,
    }
    assert normalize_slot("percent", "120 percent") is None
    assert normalize_slot("duration", "sometime") is None

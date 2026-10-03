from __future__ import annotations

import numpy as np
import pytest

from alfred.actions import MockActionExecutor
from alfred.config import load_config
from alfred.contracts import ActionResult, ApplicationState, CaptureResult, Decision, VcmResult
from alfred.message_follow_up import MessageFollowUp
from alfred.playback import NullFeedbackPlayer
from alfred.state_machine import AlfredApplication
from alfred.vcm import MockVcm


class FakeWakeDetector:
    def reset(self) -> None:
        pass


class FakeCapture:
    def __init__(self, waveform: np.ndarray | None) -> None:
        self.waveform = waveform

    def capture(self, read_frame, on_speech_start=None) -> CaptureResult:
        del read_frame
        if self.waveform is not None and on_speech_start is not None:
            on_speech_start()
        return CaptureResult(self.waveform, "fixture")


class FakeMicrophone:
    def __init__(self) -> None:
        self.clear_count = 0
        self.start_count = 0
        self.stop_count = 0

    def start(self) -> None:
        self.start_count += 1

    def stop(self) -> None:
        self.stop_count += 1

    def clear(self) -> None:
        self.clear_count += 1

    def read(self, timeout=1.0):
        del timeout
        return np.zeros(480, dtype=np.int16)


def build_app(
    result: VcmResult,
    waveform: np.ndarray | None = None,
    actions=None,
    clarification_capture=None,
    slot_clarifier=None,
    speech_recognizer=None,
    message_follow_up=None,
    weather_asr_enabled=None,
):
    config = load_config()
    feedback = NullFeedbackPlayer(config.feedback["announcements"])
    actions = actions or MockActionExecutor()
    app = AlfredApplication(
        config=config,
        wake_detector=FakeWakeDetector(),
        command_capture=FakeCapture(
            np.ones(1600, dtype=np.float32) if waveform is None else waveform
        ),
        clarification_capture=clarification_capture,
        slot_clarifier=slot_clarifier,
        speech_recognizer=speech_recognizer,
        message_follow_up=message_follow_up,
        weather_asr_enabled=weather_asr_enabled,
        vcm=MockVcm(result),
        feedback=feedback,
        actions=actions,
        sleep=lambda seconds: None,
    )
    app.transition(ApplicationState.IDLE)
    return app, feedback, actions, FakeMicrophone()


def test_primary_flow_announces_before_execution_and_returns_idle() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "LIGHT_ON", 0.99, {}, 0.99)
    )
    assert app.process_wake_event("hey_alfred", microphone)
    assert len(actions.history) == 1
    assert feedback.events == [
        "sound:listening_start",
        "sound:listening_end",
        "announce:LIGHT_ON",
        "response:Mock action completed. The lights would turn on.",
        "sound:action_success",
    ]
    assert app.transitions.index(ApplicationState.ANNOUNCING_ACTION) < app.transitions.index(
        ApplicationState.EXECUTING_ACTION
    )
    assert app.state is ApplicationState.IDLE


def test_unsupported_flow_never_executes() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.UNSUPPORTED, None, 0.2, {}, 0.1)
    )
    app.process_wake_event("hey_alfred", microphone)
    assert actions.history == []
    assert feedback.events[-1] == "sound:unsupported"


def test_non_media_command_restores_music_after_response() -> None:
    actions = MockActionExecutor()
    actions.spotify.playback_state = "PLAYING"
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "LIGHT_ON", 0.99, {}, 0.99), actions=actions
    )
    app.process_wake_event("hey_alfred", microphone)
    assert actions.spotify.playback_state == "PLAYING"
    assert actions.spotify.commands == ["interaction_pause", "interaction_complete"]
    assert feedback.events.index("response:Mock action completed. The lights would turn on.") < len(
        feedback.events
    )


def test_unexpected_interaction_failure_restores_music_before_propagating() -> None:
    class FailingFeedback(NullFeedbackPlayer):
        def play(self, event: str) -> None:
            if event == "listening_start":
                raise RuntimeError("fixture playback failure")
            super().play(event)

    actions = MockActionExecutor()
    actions.spotify.playback_state = "PLAYING"
    app, _, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "LIGHT_ON", 0.99, {}, 0.99), actions=actions
    )
    app.feedback = FailingFeedback(app.config.feedback["announcements"])

    with pytest.raises(RuntimeError, match="fixture playback failure"):
        app.process_wake_event("hey_alfred", microphone)

    assert actions.spotify.playback_state == "PLAYING"
    assert actions.spotify.commands == ["interaction_pause", "interaction_complete"]


def test_pause_intent_leaves_music_paused_after_feedback() -> None:
    actions = MockActionExecutor()
    actions.spotify.playback_state = "PLAYING"
    app, _, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "PAUSE", 0.99, {}, 0.99), actions=actions
    )
    app.process_wake_event("hey_alfred", microphone)
    assert actions.spotify.playback_state == "PAUSED"
    assert actions.spotify.commands == ["interaction_pause", "pause", "interaction_complete"]


def test_spoken_result_does_not_play_a_redundant_success_tone() -> None:
    app, feedback, _, microphone = build_app(
        VcmResult(Decision.EXECUTE, "LIGHT_ON", 0.99, {}, 0.99)
    )
    feedback.spoken_responses = True
    app.process_wake_event("hey_alfred", microphone)
    assert "response:Mock action completed. The lights would turn on." in feedback.events
    assert "sound:action_success" not in feedback.events


def test_call_releases_microphone_until_mock_call_finishes() -> None:
    app, _, _, microphone = build_app(VcmResult(Decision.EXECUTE, "CALL", 0.99, {}, 0.99))
    app.process_wake_event("hey_alfred", microphone)
    assert microphone.stop_count == 1
    assert microphone.start_count == 1
    assert ApplicationState.IN_CALL in app.transitions


def test_call_does_not_execute_when_microphone_cannot_be_released() -> None:
    class StopFailureMicrophone(FakeMicrophone):
        def stop(self) -> None:
            raise RuntimeError("fixture stop failure")

    app, feedback, actions, _ = build_app(VcmResult(Decision.EXECUTE, "CALL", 0.99, {}, 0.99))
    microphone = StopFailureMicrophone()
    app.process_wake_event("hey_alfred", microphone)
    assert actions.history == []
    assert "sound:action_failure" in feedback.events
    assert app.state is ApplicationState.IDLE


class FailingActionExecutor(MockActionExecutor):
    def execute(self, request):
        self.history.append(request)
        return ActionResult(
            request_id=request.request_id,
            status="failure",
            message="I'm unable to retrieve the weather right now.",
            handler=request.handler,
            duration_ms=1.0,
        )


def test_expected_backend_failure_is_spoken_and_uses_failure_tone_as_fallback() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "WEATHER", 0.99, {}, 0.99),
        actions=FailingActionExecutor(),
    )
    app.process_wake_event("hey_alfred", microphone)
    assert len(actions.history) == 1
    assert "response:I'm unable to retrieve the weather right now." in feedback.events
    assert feedback.events[-1] == "sound:action_failure"


def test_spoken_backend_failure_does_not_add_a_failure_earcon_after_speech() -> None:
    app, feedback, _, microphone = build_app(
        VcmResult(Decision.EXECUTE, "WEATHER", 0.99, {}, 0.99),
        actions=FailingActionExecutor(),
    )
    feedback.spoken_responses = True
    app.process_wake_event("hey_alfred", microphone)
    assert "response:I'm unable to retrieve the weather right now." in feedback.events
    assert "sound:action_failure" not in feedback.events


def test_easter_eggs_play_in_order_and_wrap_without_executing() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "LIGHT_ON", 0.99, {}, 0.99)
    )
    for _ in range(6):
        assert not app.process_wake_event("im_batman", microphone)
    assert actions.history == []
    assert feedback.events == [
        "sound:easter_egg_1",
        "sound:easter_egg_2",
        "sound:easter_egg_3",
        "sound:easter_egg_4",
        "sound:easter_egg_5",
        "sound:easter_egg_1",
    ]
    assert app.state is ApplicationState.IDLE
    assert actions.media_pause_requests == 6
    assert actions.spotify.commands == [
        item for _ in range(6) for item in ("interaction_pause", "interaction_complete")
    ]


class FakeClarifier:
    def __init__(self, transcript="ten minutes", accepted=True) -> None:
        self.transcript = transcript
        self.accepted = accepted

    def required_slot(self, result):
        del result
        return "duration"

    def prompt(self, intent):
        assert intent == "TIMER"
        return "How long should I set the timer for?"

    def transcribe(self, waveform):
        assert waveform is not None
        return self.transcript

    def apply(self, result, transcript):
        if not self.accepted:
            return None
        return VcmResult(
            result.decision,
            result.intent,
            result.intent_confidence,
            {"duration": {"surface": transcript, "seconds": 600}},
            result.in_scope_score,
            {"duration": 1.0},
        )


def test_missing_slot_uses_one_clarification_turn_before_execution() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "TIMER", 0.99, {}, 0.99),
        clarification_capture=FakeCapture(np.ones(1600, dtype=np.float32)),
        slot_clarifier=FakeClarifier(),
    )
    app.process_wake_event("hey_alfred", microphone)
    assert len(actions.history) == 1
    assert actions.history[0].slots["duration"]["seconds"] == 600
    assert "response:How long should I set the timer for?" in feedback.events
    assert ApplicationState.TRANSCRIBING_SLOT in app.transitions
    assert ApplicationState.VALIDATING_SLOT in app.transitions


def test_invalid_clarification_cancels_without_executing() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "TIMER", 0.99, {}, 0.99),
        clarification_capture=FakeCapture(np.ones(1600, dtype=np.float32)),
        slot_clarifier=FakeClarifier(accepted=False),
    )
    app.process_wake_event("hey_alfred", microphone)
    assert actions.history == []
    assert "response:I couldn't understand that value, so I won't execute it." in feedback.events
    assert app.state is ApplicationState.IDLE


def test_vcm_only_mode_rejects_a_missing_required_slot_without_asr() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "TIMER", 0.99, {}, 0.99)
    )

    app.process_wake_event("hey_alfred", microphone)

    assert actions.history == []
    assert "sound:low_confidence" in feedback.events
    assert ApplicationState.REQUESTING_SLOT not in app.transitions
    assert ApplicationState.TRANSCRIBING_SLOT not in app.transitions


class FakeSpeechRecognizer:
    def __init__(self, transcripts: list[str]) -> None:
        self.transcripts = iter(transcripts)

    def transcribe(self, waveform: np.ndarray) -> str:
        assert waveform is not None
        return next(self.transcripts)


def test_weather_collects_location_after_vcm_inference() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "WEATHER", 0.99, {}, 0.99),
        actions=FailingActionExecutor(),
        clarification_capture=FakeCapture(np.ones(1600, dtype=np.float32)),
        speech_recognizer=FakeSpeechRecognizer(["  Makati   City  "]),
    )

    app.process_wake_event("hey_alfred", microphone)

    assert len(actions.history) == 1
    assert actions.history[0].payload == {"location_query": "Makati City"}
    assert "response:Which location should I check?" in feedback.events
    assert ApplicationState.REQUESTING_WEATHER_LOCATION in app.transitions
    assert ApplicationState.CAPTURING_WEATHER_LOCATION in app.transitions
    assert ApplicationState.TRANSCRIBING_WEATHER_LOCATION in app.transitions


def test_weather_uses_default_location_when_follow_up_capture_fails() -> None:
    app, _, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "WEATHER", 0.99, {}, 0.99),
        actions=FailingActionExecutor(),
        clarification_capture=FakeCapture(None),
        speech_recognizer=FakeSpeechRecognizer([]),
    )

    app.process_wake_event("hey_alfred", microphone)

    assert len(actions.history) == 1
    assert actions.history[0].payload == {}


def test_vcm_only_weather_uses_the_configured_default_without_a_follow_up() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "WEATHER", 0.99, {}, 0.99),
        actions=FailingActionExecutor(),
    )

    app.process_wake_event("hey_alfred", microphone)

    assert len(actions.history) == 1
    assert actions.history[0].payload == {}
    assert "response:Which location should I check?" not in feedback.events
    assert ApplicationState.REQUESTING_WEATHER_LOCATION not in app.transitions


def test_weather_asr_can_be_disabled_even_when_an_asr_recognizer_is_available() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "WEATHER", 0.99, {}, 0.99),
        actions=FailingActionExecutor(),
        clarification_capture=FakeCapture(np.ones(1600, dtype=np.float32)),
        speech_recognizer=FakeSpeechRecognizer(["Makati"]),
        weather_asr_enabled=False,
    )

    app.process_wake_event("hey_alfred", microphone)

    assert actions.history[0].payload == {}
    assert "response:Which location should I check?" not in feedback.events
    assert ApplicationState.REQUESTING_WEATHER_LOCATION not in app.transitions


def test_message_collects_recipient_and_content_after_vcm_inference() -> None:
    config = load_config()
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "MESSAGE", 0.99, {}, 0.99),
        clarification_capture=FakeCapture(np.ones(1600, dtype=np.float32)),
        speech_recognizer=FakeSpeechRecognizer(["dad", "I will be home at six"]),
        message_follow_up=MessageFollowUp(config),
    )

    app.process_wake_event("hey_alfred", microphone)

    assert len(actions.history) == 1
    assert actions.history[0].slots == {}
    assert actions.history[0].payload == {
        "recipient": "Dad",
        "message": "I will be home at six",
    }
    assert "response:Who should I address the message to?" in feedback.events
    assert "response:What message should I send?" in feedback.events
    assert ApplicationState.TRANSCRIBING_MESSAGE_RECIPIENT in app.transitions
    assert ApplicationState.TRANSCRIBING_MESSAGE_CONTENT in app.transitions
    assert ApplicationState.VALIDATING_MESSAGE in app.transitions
    assert app.state is ApplicationState.IDLE


def test_invalid_message_recipient_sends_nothing() -> None:
    config = load_config()
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "MESSAGE", 0.99, {}, 0.99),
        clarification_capture=FakeCapture(np.ones(1600, dtype=np.float32)),
        speech_recognizer=FakeSpeechRecognizer([""]),
        message_follow_up=MessageFollowUp(config),
    )

    app.process_wake_event("hey_alfred", microphone)

    assert actions.history == []
    assert "response:I couldn't prepare your message, so nothing was sent." in feedback.events
    assert "response:What message should I send?" not in feedback.events
    assert app.state is ApplicationState.IDLE


def test_vcm_only_message_uses_action_layer_fallback_without_asr() -> None:
    app, feedback, actions, microphone = build_app(
        VcmResult(Decision.EXECUTE, "MESSAGE", 0.99, {}, 0.99)
    )

    app.process_wake_event("hey_alfred", microphone)

    assert len(actions.history) == 1
    assert actions.history[0].payload == {}
    assert "response:I couldn't prepare your message, so nothing was sent." not in feedback.events
    assert ApplicationState.REQUESTING_MESSAGE_RECIPIENT not in app.transitions
    assert ApplicationState.TRANSCRIBING_MESSAGE_CONTENT not in app.transitions

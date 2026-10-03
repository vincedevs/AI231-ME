from __future__ import annotations

import json

import numpy as np
import pytest

from alfred.actions import MockActionExecutor
from alfred.benchmark_logging import BenchmarkEventLogger
from alfred.config import load_config
from alfred.contracts import ApplicationState, Decision, VcmResult
from alfred.playback import NullFeedbackPlayer
from alfred.state_machine import AlfredApplication
from alfred.vcm import MockVcm


class FakeWakeDetector:
    def reset(self) -> None:
        pass


class FakeCapture:
    def __init__(self, waveform: np.ndarray) -> None:
        self.waveform = waveform

    def capture(self, read_frame, on_speech_start=None):
        del read_frame
        if on_speech_start is not None:
            on_speech_start()
        return type("Capture", (), {"waveform": self.waveform, "reason": "fixture"})()


class FakeMicrophone:
    def clear(self) -> None:
        pass

    def read(self, timeout=1.0):
        del timeout
        return np.zeros(480, dtype=np.int16)


class FailIfUsed:
    """Fail a test if benchmark mode touches a suppressed component."""

    def __getattr__(self, name):
        raise AssertionError(f"benchmark mode unexpectedly used {name}")


def build_benchmark_app(
    tmp_path,
    result: VcmResult,
) -> tuple[AlfredApplication, BenchmarkEventLogger, FakeMicrophone]:
    config = load_config()
    logger = BenchmarkEventLogger(tmp_path)

    def fail_sleep(_seconds: float) -> None:
        raise AssertionError("benchmark mode unexpectedly entered the feedback cooldown")

    app = AlfredApplication(
        config=config,
        wake_detector=FakeWakeDetector(),
        command_capture=FakeCapture(np.ones(1600, dtype=np.float32)),
        clarification_capture=FailIfUsed(),
        slot_clarifier=FailIfUsed(),
        speech_recognizer=FailIfUsed(),
        message_follow_up=FailIfUsed(),
        vcm=MockVcm(result),
        feedback=FailIfUsed(),
        actions=FailIfUsed(),
        benchmark_logger=logger,
        benchmark_mode=True,
        sleep=fail_sleep,
    )
    app.transition(ApplicationState.IDLE)
    return app, logger, FakeMicrophone()


def read_events(logger: BenchmarkEventLogger) -> list[dict[str, object]]:
    logger.close()
    return [json.loads(line) for line in logger.path.read_text(encoding="utf-8").splitlines()]


def test_logger_uses_benchmark_contract_and_keeps_rejections_out_of_scope(tmp_path) -> None:
    logger = BenchmarkEventLogger(tmp_path)
    logger.wake()
    logger.command(
        VcmResult(
            Decision.EXECUTE,
            "TIMER",
            0.91,
            {"duration": {"surface": "30 seconds"}},
            0.98,
        ),
        infer_ms=84.2,
        audio_ms=1500,
    )
    logger.command(
        VcmResult(Decision.UNSUPPORTED, "LIGHT_ON", 0.95, {}, 0.12),
        infer_ms=10,
        audio_ms=800,
    )
    wake, timer, rejected = read_events(logger)
    assert wake == {"event": "wake", "wake_word": "hey_alfred"}
    assert timer["intent"] == "TIMER"
    assert timer["slot"] == "30 seconds"
    assert timer["infer_ms"] == 84.2
    assert timer["audio_ms"] == 1500.0
    assert rejected["intent"] == "OUT_OF_SCOPE"
    assert rejected["slot"] == ""


def test_primary_wake_writes_one_wake_and_one_command_with_vcm_timing(tmp_path) -> None:
    config = load_config()
    logger = BenchmarkEventLogger(tmp_path)
    app = AlfredApplication(
        config=config,
        wake_detector=FakeWakeDetector(),
        command_capture=FakeCapture(np.ones(1600, dtype=np.float32)),
        vcm=MockVcm(VcmResult(Decision.EXECUTE, "LIGHT_ON", 0.99, {}, 0.99)),
        feedback=NullFeedbackPlayer(config.feedback["announcements"]),
        actions=MockActionExecutor(),
        benchmark_logger=logger,
        sleep=lambda _: None,
    )
    app.transition(ApplicationState.IDLE)
    assert app.process_wake_event("hey_alfred", FakeMicrophone())
    events = read_events(logger)
    assert [event["event"] for event in events] == ["wake", "command"]
    assert events[1]["intent"] == "LIGHT_ON"
    assert events[1]["audio_ms"] == 100.0
    assert isinstance(events[1]["infer_ms"], float)


def test_easter_egg_never_creates_a_benchmark_wake_or_command(tmp_path) -> None:
    config = load_config()
    logger = BenchmarkEventLogger(tmp_path)
    app = AlfredApplication(
        config=config,
        wake_detector=FakeWakeDetector(),
        command_capture=FakeCapture(np.ones(1600, dtype=np.float32)),
        vcm=MockVcm(VcmResult(Decision.EXECUTE, "LIGHT_ON", 0.99, {}, 0.99)),
        feedback=NullFeedbackPlayer(config.feedback["announcements"]),
        actions=MockActionExecutor(),
        benchmark_logger=logger,
        sleep=lambda _: None,
    )
    app.transition(ApplicationState.IDLE)
    assert not app.process_wake_event("im_batman", FakeMicrophone())
    assert read_events(logger) == []


def test_benchmark_mode_logs_one_execute_result_and_suppresses_post_inference_work(
    tmp_path,
) -> None:
    app, logger, microphone = build_benchmark_app(
        tmp_path,
        VcmResult(
            Decision.EXECUTE,
            "TIMER",
            0.98,
            {"duration": {"surface": "thirty seconds"}},
            0.99,
        ),
    )

    assert app.process_wake_event("hey_alfred", microphone)

    events = read_events(logger)
    assert [event["event"] for event in events] == ["wake", "command"]
    assert events[1]["intent"] == "TIMER"
    assert events[1]["slot"] == "thirty seconds"
    assert app.state is ApplicationState.IDLE
    assert ApplicationState.CHECKING_POLICY not in app.transitions


def test_benchmark_mode_does_not_start_message_follow_up(tmp_path) -> None:
    app, logger, microphone = build_benchmark_app(
        tmp_path,
        VcmResult(Decision.EXECUTE, "MESSAGE", 0.98, {}, 0.99),
    )

    assert app.process_wake_event("hey_alfred", microphone)

    assert [event["event"] for event in read_events(logger)] == ["wake", "command"]
    assert app.state is ApplicationState.IDLE


@pytest.mark.parametrize("decision", [Decision.LOW_CONFIDENCE, Decision.UNSUPPORTED])
def test_benchmark_mode_records_rejections_as_out_of_scope(tmp_path, decision) -> None:
    app, logger, microphone = build_benchmark_app(
        tmp_path,
        VcmResult(decision, "LIGHT_ON", 0.51, {}, 0.45),
    )

    assert app.process_wake_event("hey_alfred", microphone)

    events = read_events(logger)
    assert len(events) == 2
    assert events[1]["intent"] == "OUT_OF_SCOPE"
    assert events[1]["decision"] == decision.value


def test_benchmark_mode_ignores_batman_without_feedback_or_logging(tmp_path) -> None:
    app, logger, microphone = build_benchmark_app(
        tmp_path,
        VcmResult(Decision.EXECUTE, "LIGHT_ON", 0.99, {}, 0.99),
    )

    assert not app.process_wake_event("im_batman", microphone)

    assert read_events(logger) == []
    assert app.state is ApplicationState.IDLE

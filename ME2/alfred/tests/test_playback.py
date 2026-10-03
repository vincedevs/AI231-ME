from __future__ import annotations

import pytest

from alfred.playback import FeedbackPlayer, announcement_text


class FakeTts:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def speak(self, text: str) -> dict[str, float]:
        self.messages.append(text)
        return {"generation_seconds": 0.1, "audio_seconds": 1.0, "real_time_factor": 0.1}


def test_announcement_includes_the_normalized_slot_surface() -> None:
    assert (
        announcement_text(
            "Setting a timer for {duration}.",
            {"duration": {"surface": "ten minutes", "seconds": 600}},
        )
        == "Setting a timer for ten minutes."
    )


def test_announcement_rejects_a_missing_required_placeholder() -> None:
    with pytest.raises(ValueError, match="missing slot"):
        announcement_text("Setting a timer for {duration}.", {})


def test_feedback_speaks_announcements_and_action_results() -> None:
    from alfred.config import load_config

    tts = FakeTts()
    feedback = FeedbackPlayer(load_config(), tts=tts)
    assert feedback.spoken_responses
    assert feedback.announce("LIGHT_ON", {}) == "Turning on the lights."
    assert feedback.respond("Mock action completed.") == "Mock action completed."
    assert tts.messages == ["Turning on the lights.", "Mock action completed."]


def test_multiline_response_is_spoken_in_segments_with_bounded_pauses() -> None:
    from alfred.config import load_config

    tts = FakeTts()
    pauses: list[float] = []
    feedback = FeedbackPlayer(load_config(), tts=tts, sleep=pauses.append)
    message = "You have 2 reminders.\nReminder 1. Buy milk.\nReminder 2. Call Dad."

    assert feedback.respond(message) == message
    assert tts.messages == [
        "You have 2 reminders.",
        "Reminder 1. Buy milk.",
        "Reminder 2. Call Dad.",
    ]
    assert pauses == [0.4, 0.4]

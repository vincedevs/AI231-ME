from __future__ import annotations

import pytest

from alfred.clarification import SlotClarifier, normalize_clarified_slot
from alfred.config import load_config
from alfred.contracts import Decision, VcmResult
from alfred.policy import normalize_slot


class FakeRecognizer:
    def __init__(self, transcript: str = "") -> None:
        self.transcript = transcript

    def transcribe(self, waveform) -> str:
        del waveform
        return self.transcript


@pytest.mark.parametrize(
    ("intent", "slot", "transcript", "normalized_key", "expected"),
    [
        ("TIMER", "duration", "five seconds.", "seconds", 5),
        ("TIMER", "duration", "ten minutes", "seconds", 600),
        ("ALARM", "time", "4 PM.", "text", "4 pm"),
        ("ALARM", "time", "seven thirty am", "text", "seven thirty am"),
        ("TEMPERATURE", "degrees", "twenty four degrees", "value", 24.0),
        ("BRIGHTNESS", "percent", "sixty five percent", "value", 65.0),
        ("COLOR", "color", "warm white", "text", "warm white"),
        ("CREATE_REMINDER", "task", "submit my thesis", "text", "submit my thesis"),
    ],
)
def test_clarified_slot_normalization_covers_six_canonical_slots(
    intent: str,
    slot: str,
    transcript: str,
    normalized_key: str,
    expected,
) -> None:
    normalized = normalize_clarified_slot(intent, slot, transcript, load_config())
    assert normalized is not None
    assert normalized[normalized_key] == expected
    assert normalized["source"] == "clarification_asr"


@pytest.mark.parametrize(
    ("intent", "slot", "transcript"),
    [
        ("TIMER", "duration", "eventually"),
        ("TIMER", "duration", "for minutes"),
        ("ALARM", "time", "some time tomorrow"),
        ("ALARM", "time", "for PM"),
        ("TEMPERATURE", "degrees", "fifty degrees"),
        ("BRIGHTNESS", "percent", "one hundred fifty percent"),
        ("COLOR", "color", "chartreuse"),
        ("CREATE_REMINDER", "task", ""),
    ],
)
def test_invalid_clarification_never_becomes_an_action_slot(
    intent: str, slot: str, transcript: str
) -> None:
    assert normalize_clarified_slot(intent, slot, transcript, load_config()) is None


def test_clarifier_requests_missing_or_low_confidence_slots_only() -> None:
    config = load_config()
    clarifier = SlotClarifier(config, FakeRecognizer())
    missing = VcmResult(Decision.EXECUTE, "TIMER", 0.95, {}, 0.95)
    low = VcmResult(
        Decision.EXECUTE,
        "TIMER",
        0.95,
        {"duration": {"surface": "ten minutes"}},
        0.95,
        {"duration": 0.6},
    )
    accepted = VcmResult(
        Decision.EXECUTE,
        "TIMER",
        0.95,
        {"duration": {"surface": "ten minutes"}},
        0.95,
        {"duration": 0.9},
    )
    slotless = VcmResult(Decision.EXECUTE, "MESSAGE", 0.95, {}, 0.95)
    uncertain = VcmResult(Decision.LOW_CONFIDENCE, "TIMER", 0.5, {}, 0.95)
    assert clarifier.required_slot(missing) == "duration"
    assert clarifier.required_slot(low) == "duration"
    assert clarifier.required_slot(accepted) is None
    assert clarifier.required_slot(slotless) is None
    assert clarifier.required_slot(uncertain) is None


def test_clarification_replaces_only_the_expected_slot() -> None:
    config = load_config()
    clarifier = SlotClarifier(config, FakeRecognizer())
    original = VcmResult(Decision.EXECUTE, "CREATE_REMINDER", 0.98, {}, 0.97)
    recovered = clarifier.apply(original, "submit the experiment report")
    assert recovered is not None
    assert recovered.slots["task"] == {
        "surface": "submit the experiment report",
        "text": "submit the experiment report",
        "source": "clarification_asr",
    }


def test_reminder_clarification_removes_artifacts_without_rewriting_words() -> None:
    normalized = normalize_clarified_slot(
        "CREATE_REMINDER",
        "task",
        "by milk dot",
        load_config(),
    )
    assert normalized == {
        "surface": "by milk",
        "text": "by milk",
        "source": "clarification_asr",
    }
    assert normalize_slot("task", {"surface": "by milk dot"}) == {
        "surface": "by milk",
        "text": "by milk",
    }

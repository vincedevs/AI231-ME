from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any

from .config import AlfredConfig
from .contracts import Decision, VcmResult
from .policy import normalize_slot
from .scheduling import SchedulingError, next_alarm_time
from .speech_recognition import (
    MoonshineRecognizer,
    SpeechRecognitionError,
    build_speech_recognizer,
    clean_asr_transcript,
)

LOGGER = logging.getLogger(__name__)


def normalize_clarified_slot(
    intent: str,
    slot_name: str,
    transcript: str,
    config: AlfredConfig,
) -> dict[str, Any] | None:
    """Validate a one-turn ASR answer against the canonical slot semantics."""
    text = clean_asr_transcript(transcript)
    if not text:
        return None
    if slot_name == "task":
        maximum = int(config.document["actions"]["reminders"]["max_task_characters"])
        if len(text) > maximum:
            return None
        return {"surface": text, "text": text, "source": "clarification_asr"}

    normalized = normalize_slot(slot_name, {"surface": text})
    if normalized is None:
        return None
    wayne_manor = config.document["actions"]["wayne_manor"]
    if intent == "TEMPERATURE":
        value = float(normalized["value"])
        if not (
            float(wayne_manor["minimum_temperature_degrees"])
            <= value
            <= float(wayne_manor["maximum_temperature_degrees"])
        ):
            return None
    elif intent == "COLOR":
        supported = {str(item) for item in wayne_manor["fallback_colors"]}
        if str(normalized["text"]) not in supported:
            return None
    elif intent == "TIMER":
        maximum = int(config.document["actions"]["scheduler"]["maximum_timer_seconds"])
        if int(normalized["seconds"]) > maximum:
            return None
    elif intent == "ALARM":
        try:
            next_alarm_time(str(normalized["text"]))
        except (SchedulingError, ValueError):
            return None
    normalized["source"] = "clarification_asr"
    return normalized


class SlotClarifier:
    """Request one explicit follow-up only for the six canonical slot intents."""

    def __init__(self, config: AlfredConfig, recognizer: MoonshineRecognizer) -> None:
        self.config = config
        self.recognizer = recognizer
        values = config.document["clarification"]
        self.minimum_confidence = float(values["minimum_slot_confidence"])
        self.prompts = {str(key): str(value) for key, value in values["prompts"].items()}

    def required_slot(self, result: VcmResult) -> str | None:
        if result.decision is not Decision.EXECUTE or result.intent not in self.prompts:
            return None
        slot_name = self.config.slot_by_intent.get(str(result.intent))
        if slot_name is None:
            return None
        normalized = normalize_clarified_slot(
            str(result.intent),
            slot_name,
            str(result.slots.get(slot_name, {}).get("surface", ""))
            if isinstance(result.slots.get(slot_name), dict)
            else str(result.slots.get(slot_name, "")),
            self.config,
        )
        confidence = float(result.slot_confidences.get(slot_name, 0.0))
        if normalized is None or confidence < self.minimum_confidence:
            return slot_name
        return None

    def prompt(self, intent: str) -> str:
        return self.prompts[intent]

    def transcribe(self, waveform: Any) -> str:
        started = time.perf_counter()
        try:
            return self.recognizer.transcribe(waveform)
        finally:
            LOGGER.info(
                "clarification_asr_duration_ms=%.3f",
                (time.perf_counter() - started) * 1000,
            )

    def apply(self, result: VcmResult, transcript: str) -> VcmResult | None:
        if result.intent is None:
            return None
        slot_name = self.config.slot_by_intent.get(result.intent)
        if slot_name is None:
            return result
        normalized = normalize_clarified_slot(result.intent, slot_name, transcript, self.config)
        if normalized is None:
            return None
        slots = dict(result.slots)
        slots[slot_name] = normalized
        confidences = dict(result.slot_confidences)
        # ASR does not provide a calibrated VCM slot posterior. Remove the old
        # VCM confidence and preserve provenance through the source tag instead.
        confidences.pop(slot_name, None)
        return replace(result, slots=slots, slot_confidences=confidences)


def build_slot_clarifier(
    config: AlfredConfig,
    recognizer: MoonshineRecognizer | None = None,
) -> SlotClarifier | None:
    values = config.document["clarification"]
    if not bool(values["enabled"]):
        return None
    recognizer = recognizer or build_speech_recognizer(config)
    if recognizer is None:
        return None
    return SlotClarifier(config, recognizer)


__all__ = [
    "SlotClarifier",
    "SpeechRecognitionError",
    "build_slot_clarifier",
    "normalize_clarified_slot",
]

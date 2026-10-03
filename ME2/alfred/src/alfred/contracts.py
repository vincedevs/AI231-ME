from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Decision(StrEnum):
    EXECUTE = "execute"
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED = "unsupported"


class ApplicationState(StrEnum):
    STARTING = "starting"
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    PLAYING_LISTENING_CUE = "playing_listening_cue"
    WAITING_FOR_SPEECH = "waiting_for_speech"
    CAPTURING_COMMAND = "capturing_command"
    INFERRING = "inferring"
    REQUESTING_SLOT = "requesting_slot"
    PLAYING_CLARIFICATION_CUE = "playing_clarification_cue"
    WAITING_FOR_SLOT = "waiting_for_slot"
    CAPTURING_SLOT = "capturing_slot"
    TRANSCRIBING_SLOT = "transcribing_slot"
    VALIDATING_SLOT = "validating_slot"
    REQUESTING_MESSAGE_RECIPIENT = "requesting_message_recipient"
    WAITING_FOR_MESSAGE_RECIPIENT = "waiting_for_message_recipient"
    CAPTURING_MESSAGE_RECIPIENT = "capturing_message_recipient"
    TRANSCRIBING_MESSAGE_RECIPIENT = "transcribing_message_recipient"
    REQUESTING_MESSAGE_CONTENT = "requesting_message_content"
    WAITING_FOR_MESSAGE_CONTENT = "waiting_for_message_content"
    CAPTURING_MESSAGE_CONTENT = "capturing_message_content"
    TRANSCRIBING_MESSAGE_CONTENT = "transcribing_message_content"
    VALIDATING_MESSAGE = "validating_message"
    REQUESTING_WEATHER_LOCATION = "requesting_weather_location"
    WAITING_FOR_WEATHER_LOCATION = "waiting_for_weather_location"
    CAPTURING_WEATHER_LOCATION = "capturing_weather_location"
    TRANSCRIBING_WEATHER_LOCATION = "transcribing_weather_location"
    CHECKING_POLICY = "checking_policy"
    ANNOUNCING_ACTION = "announcing_action"
    EXECUTING_ACTION = "executing_action"
    IN_CALL = "in_call"
    PLAYING_FEEDBACK = "playing_feedback"
    EASTER_EGG = "easter_egg"
    COOLDOWN = "cooldown"
    STOPPING = "stopping"


@dataclass(frozen=True)
class VcmResult:
    decision: Decision
    intent: str | None
    intent_confidence: float
    slots: dict[str, Any]
    in_scope_score: float | None = None
    slot_confidences: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> VcmResult:
        decision = Decision(str(value["decision"]))
        intent = value.get("intent")
        confidence = float(value["intent_confidence"])
        in_scope = value.get("in_scope_score")
        slots = value.get("slots", {})
        slot_confidences = value.get("slot_confidences", {})
        if intent is not None and not isinstance(intent, str):
            raise ValueError("VCM intent must be a string or null")
        if not 0 <= confidence <= 1:
            raise ValueError("VCM intent confidence must be between 0 and 1")
        if in_scope is not None and not 0 <= float(in_scope) <= 1:
            raise ValueError("VCM in-scope score must be between 0 and 1")
        if not isinstance(slots, dict) or not isinstance(slot_confidences, dict):
            raise TypeError("VCM slots and slot confidences must be objects")
        parsed_slot_confidences = {str(key): float(item) for key, item in slot_confidences.items()}
        if any(not 0 <= item <= 1 for item in parsed_slot_confidences.values()):
            raise ValueError("VCM slot confidences must be between 0 and 1")
        return cls(
            decision=decision,
            intent=intent,
            intent_confidence=confidence,
            slots=dict(slots),
            in_scope_score=None if in_scope is None else float(in_scope),
            slot_confidences=parsed_slot_confidences,
        )


@dataclass(frozen=True)
class ActionRequest:
    request_id: str
    schema_version: str
    intent: str
    handler: str
    slots: dict[str, Any]
    payload: dict[str, Any]
    intent_confidence: float
    in_scope_score: float | None
    created_monotonic: float

    @classmethod
    def create(
        cls,
        *,
        schema_version: str,
        intent: str,
        handler: str,
        slots: dict[str, Any],
        payload: dict[str, Any] | None = None,
        intent_confidence: float,
        in_scope_score: float | None,
    ) -> ActionRequest:
        return cls(
            request_id=str(uuid.uuid4()),
            schema_version=schema_version,
            intent=intent,
            handler=handler,
            slots=slots,
            payload={} if payload is None else dict(payload),
            intent_confidence=intent_confidence,
            in_scope_score=in_scope_score,
            created_monotonic=time.monotonic(),
        )


@dataclass(frozen=True)
class ActionResult:
    request_id: str
    status: str
    message: str
    handler: str
    duration_ms: float


@dataclass(frozen=True)
class PolicyOutcome:
    decision: Decision
    reason: str
    action_request: ActionRequest | None = None


@dataclass(frozen=True)
class CaptureResult:
    waveform: Any | None
    reason: str
    speech_seconds: float = 0.0

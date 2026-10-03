from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INTENTS = (
    "PLAY_MUSIC",
    "WEATHER",
    "TIME",
    "LIGHT_ON",
    "LIGHT_OFF",
    "PAUSE",
    "STOP",
    "NEXT",
    "VOLUME_UP",
    "VOLUME_DOWN",
    "CALL",
    "MESSAGE",
    "LIST_REMINDERS",
    "TIMER",
    "ALARM",
    "TEMPERATURE",
    "BRIGHTNESS",
    "COLOR",
    "CREATE_REMINDER",
)

SLOT_BY_INTENT = {
    "TIMER": "duration",
    "ALARM": "time",
    "TEMPERATURE": "degrees",
    "BRIGHTNESS": "percent",
    "COLOR": "color",
    "CREATE_REMINDER": "task",
}


def load_schema(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    names = tuple(item["name"] for item in document["intents"])
    slots = {item["name"]: item["slot"] for item in document["intents"] if item["slot"] is not None}
    if names != INTENTS:
        raise ValueError("configs/vcm_schema.json does not match the frozen intent order")
    if slots != SLOT_BY_INTENT:
        raise ValueError("configs/vcm_schema.json does not match the frozen slot schema")
    return document


def validate_prepared_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "sample_id",
        "audio_path",
        "source_dataset",
        "source_id",
        "speaker_id",
        "recording_group_id",
        "utterance_group_id",
        "transcript",
        "canonical_intent",
        "slots",
        "supported",
        "is_synthetic",
        "mapping_rule",
    }
    missing = sorted(required - set(record))
    if missing:
        errors.append(f"missing fields: {missing}")
        return errors

    supported = bool(record["supported"])
    intent = record["canonical_intent"]
    if supported and intent not in INTENTS:
        errors.append(f"unsupported canonical intent: {intent!r}")
        return errors
    if not supported and intent is not None:
        errors.append("unsupported records must have canonical_intent=null")
        return errors

    slots = record["slots"]
    if not isinstance(slots, dict):
        errors.append("slots must be an object")
        return errors
    if not supported:
        if slots:
            errors.append("unsupported records must not contain canonical slots")
        return errors
    expected = SLOT_BY_INTENT.get(intent)
    if expected is None and slots:
        errors.append(f"fixed intent {intent} must not contain slots")
    if expected is not None and slots:
        if set(slots) != {expected}:
            errors.append(f"{intent} may contain only the {expected!r} slot")
        else:
            slot = slots[expected]
            if not isinstance(slot, dict) or not str(slot.get("surface", "")).strip():
                errors.append(f"{intent}.{expected} requires a non-empty surface value")
    return errors

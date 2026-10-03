from __future__ import annotations

import re
import unicodedata
from typing import Any

from .config import AlfredConfig
from .contracts import ActionRequest, Decision, PolicyOutcome, VcmResult

SMALL_NUMBERS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9%']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def first_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    if match:
        return float(match.group())
    total = 0
    current = 0
    found = False
    for token in normalize_text(value).split():
        if token in SMALL_NUMBERS:
            current += SMALL_NUMBERS[token]
            found = True
        elif token in TENS:
            current += TENS[token]
            found = True
        elif token == "hundred" and found:
            current = max(1, current) * 100
        elif token == "thousand" and found:
            total += max(1, current) * 1000
            current = 0
        elif found:
            break
    return float(total + current) if found else None


def slot_surface(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return str(value.get("surface") or value.get("text") or "").strip()
    return str(value).strip() if value is not None else ""


def normalize_slot(name: str, value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict) and isinstance(value.get("normalized"), dict):
        normalized = dict(value["normalized"])
        surface = slot_surface(value)
    else:
        normalized = {}
        surface = slot_surface(value)
    clean = normalize_text(surface)
    if not clean and not normalized:
        return None

    if name == "duration":
        seconds = normalized.get("seconds")
        if seconds is None:
            number = first_number(clean)
            if number is None:
                return None
            if "hour" in clean:
                seconds = round(number * 3600)
            elif "minute" in clean:
                seconds = round(number * 60)
            elif "second" in clean:
                seconds = round(number)
            else:
                return None
        if int(seconds) <= 0:
            return None
        spoken_surface = re.sub(r"^(?:for|in)\s+", "", surface, flags=re.IGNORECASE)
        return {"surface": spoken_surface, "seconds": int(seconds)}

    if name == "time":
        text = str(normalized.get("text") or clean).strip()
        spoken_surface = re.sub(r"^(?:at|for)\s+", "", surface, flags=re.IGNORECASE)
        return {"surface": spoken_surface, "text": text} if text else None

    if name in {"degrees", "percent"}:
        number = normalized.get("value")
        if number is None:
            number = first_number(clean)
        if number is None:
            return None
        number = float(number)
        if name == "percent" and not 0 <= number <= 100:
            return None
        return {
            "surface": surface,
            "value": number,
            "unit": "percent" if name == "percent" else "degrees",
        }

    if name == "task":
        text = re.sub(r"(?:\s+(?:dot|period|full stop))$", "", clean)
        if not text:
            return None
        output = {"surface": text, "text": text}
        if isinstance(value, dict) and value.get("source") in {
            "vcm",
            "clarification_asr",
            "fallback_default",
        }:
            output["source"] = str(value["source"])
        return output

    text = str(normalized.get("text") or clean).strip()
    if not text:
        return None
    output = {"surface": surface, "text": text}
    if isinstance(value, dict) and value.get("source") in {
        "vcm",
        "clarification_asr",
        "fallback_default",
    }:
        output["source"] = str(value["source"])
    return output


def apply_policy(result: VcmResult, config: AlfredConfig) -> PolicyOutcome:
    if result.decision is Decision.UNSUPPORTED:
        return PolicyOutcome(Decision.UNSUPPORTED, "VCM classified the command as unsupported")
    if result.decision is Decision.LOW_CONFIDENCE:
        return PolicyOutcome(Decision.LOW_CONFIDENCE, "VCM confidence was below its threshold")
    if result.intent is None or result.intent not in config.intents:
        return PolicyOutcome(Decision.UNSUPPORTED, "VCM returned no canonical intent")

    minimum = float(config.document["vcm"]["minimum_intent_confidence"])
    if result.intent_confidence < minimum:
        return PolicyOutcome(
            Decision.LOW_CONFIDENCE,
            f"Intent confidence {result.intent_confidence:.3f} is below {minimum:.3f}",
        )

    enabled_intents = set(config.document["actions"]["enabled_intents"])
    if result.intent not in enabled_intents:
        return PolicyOutcome(
            Decision.UNSUPPORTED,
            f"Action execution is disabled for intent {result.intent}",
        )

    action = config.action_by_intent[result.intent]
    normalized_slots: dict[str, Any] = {}
    expected_slot = config.slot_by_intent.get(result.intent)
    if expected_slot is not None and expected_slot in result.slots:
        normalized = normalize_slot(expected_slot, result.slots[expected_slot])
        wayne_manor = config.document["actions"]["wayne_manor"]
        if result.intent == "TEMPERATURE" and normalized is not None:
            value = float(normalized["value"])
            if not (
                float(wayne_manor["minimum_temperature_degrees"])
                <= value
                <= float(wayne_manor["maximum_temperature_degrees"])
            ):
                normalized = None
        if result.intent == "COLOR" and normalized is not None:
            supported = {str(item) for item in wayne_manor["fallback_colors"]}
            if str(normalized["text"]) not in supported:
                normalized = None
        if normalized is not None:
            normalized_slots[expected_slot] = normalized
    required = tuple(action.get("required_slots", []))
    missing = [name for name in required if name not in normalized_slots]
    if missing:
        return PolicyOutcome(
            Decision.LOW_CONFIDENCE,
            f"Required slot values are missing or invalid: {', '.join(missing)}",
        )

    request = ActionRequest.create(
        schema_version=str(config.schema["schema_version"]),
        intent=result.intent,
        handler=str(action["handler"]),
        slots=normalized_slots,
        intent_confidence=result.intent_confidence,
        in_scope_score=result.in_scope_score,
    )
    return PolicyOutcome(Decision.EXECUTE, "Command passed the policy gate", request)

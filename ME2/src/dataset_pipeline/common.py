from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9%']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_id(source: str, source_id: str) -> str:
    prefix = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    return f"{prefix}_{stable_digest(source + '|' + source_id)[:20]}"


def find_span(transcript: str, surface: str) -> tuple[int | None, int | None]:
    start = transcript.lower().find(surface.lower())
    if start >= 0:
        return start, start + len(surface)

    # Some source annotations insert spaces around punctuation that the
    # verbatim transcript omits (for example ``6 : 00`` versus ``6:00``).
    # Align on alphanumeric tokens while retaining original character offsets.
    normalized_chars: list[str] = []
    original_positions: list[int] = []
    previous_was_space = True
    previous_was_digit: bool | None = None
    for original_index, original_char in enumerate(transcript):
        expanded = unicodedata.normalize("NFKC", original_char).lower()
        for char in expanded:
            if char.isalnum() or char == "%":
                is_digit = char.isdigit()
                if (
                    not previous_was_space
                    and previous_was_digit is not None
                    and is_digit != previous_was_digit
                ):
                    normalized_chars.append(" ")
                    original_positions.append(original_index)
                normalized_chars.append(char)
                original_positions.append(original_index)
                previous_was_space = False
                previous_was_digit = is_digit
            elif not previous_was_space:
                normalized_chars.append(" ")
                original_positions.append(original_index)
                previous_was_space = True
                previous_was_digit = None
    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        original_positions.pop()

    normalized_transcript = "".join(normalized_chars)
    normalized_surface = re.sub(
        r"\s+",
        " ",
        re.sub(
            r"[^a-z0-9%]+",
            " ",
            unicodedata.normalize("NFKC", surface).lower(),
        ),
    ).strip()
    normalized_surface = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", normalized_surface)
    normalized_start = normalized_transcript.find(normalized_surface)
    if normalized_start < 0 or not normalized_surface:
        return None, None
    normalized_end = normalized_start + len(normalized_surface) - 1
    return original_positions[normalized_start], original_positions[normalized_end] + 1


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


def first_number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value.replace(",", ""))
    if match:
        return float(match.group())
    tokens = normalize_text(value).replace("-", " ").split()
    total = 0
    current = 0
    found = False
    for token in tokens:
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


def normalize_slot(name: str, surface: str) -> dict[str, Any]:
    clean = normalize_text(surface)
    if name == "duration":
        normalized: dict[str, Any] = {"text": clean}
        number = first_number(clean)
        if number is not None:
            if "hour" in clean:
                normalized["seconds"] = int(number * 3600)
            elif "minute" in clean:
                normalized["seconds"] = int(number * 60)
            elif "second" in clean:
                normalized["seconds"] = int(number)
    elif name == "time":
        normalized = {"text": clean}
    elif name == "degrees":
        normalized = {"value": first_number(clean), "unit": "degrees"}
    elif name == "percent":
        normalized = {"value": first_number(clean), "unit": "percent"}
    else:
        normalized = {"text": clean}
    return normalized


def slot_value(name: str, surface: str, transcript: str) -> dict[str, Any]:
    start, end = find_span(transcript, surface)
    return {
        "surface": surface.strip(),
        "normalized": normalize_slot(name, surface),
        "char_start": start,
        "char_end": end,
    }


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]

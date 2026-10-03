"""Write the small, append-only event log consumed by ``vcm-benchmark``.

The benchmark scores one semantic VCM result per primary wake word. This
module logs the result immediately after VCM inference, before slot
clarification, policy defaults, action execution, or TTS can affect it.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import Decision, VcmResult


class BenchmarkEventLogger:
    """Append benchmark-compatible JSON Lines to one file per Alfred process."""

    def __init__(self, directory: Path | None = None) -> None:
        root = directory or Path(
            os.environ.get("ALFRED_BENCHMARK_LOG_DIR", "~/vcm_benchmark")
        ).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.path = root / f"alfred_{timestamp}_{os.getpid()}.log"
        self._stream = self.path.open("a", encoding="utf-8", buffering=1)

    def _write(self, event: dict[str, Any]) -> None:
        self._stream.write(json.dumps(event, separators=(",", ":"), sort_keys=True) + "\n")
        self._stream.flush()

    def wake(self) -> None:
        """Record a primary wake event without creating a command prediction."""
        self._write({"event": "wake", "wake_word": "hey_alfred"})

    def command(self, result: VcmResult, *, infer_ms: float, audio_ms: float) -> None:
        """Record exactly one scored semantic result for a captured command.

        A rejected argmax remains a rejection. Logging its raw top intent would
        make the benchmark count it as an unintended command rather than an
        out-of-scope response.
        """
        intent = result.intent if result.decision is Decision.EXECUTE else "OUT_OF_SCOPE"
        slot = ""
        if result.decision is Decision.EXECUTE and result.intent is not None:
            slot = _slot_surface(result.slots)
        self._write(
            {
                "event": "command",
                "intent": intent,
                "slot": slot,
                "infer_ms": round(max(0.0, float(infer_ms)), 3),
                "audio_ms": round(max(0.0, float(audio_ms)), 3),
                "decision": result.decision.value,
                "intent_confidence": round(float(result.intent_confidence), 6),
                "in_scope_score": (
                    None
                    if result.in_scope_score is None
                    else round(float(result.in_scope_score), 6)
                ),
            }
        )

    def close(self) -> None:
        self._stream.close()


def _slot_surface(slots: dict[str, Any]) -> str:
    """Return the VCM-decoded slot string, never an action-layer default."""
    for value in slots.values():
        surface = value.get("surface") if isinstance(value, dict) else value
        if surface is not None:
            return str(surface).strip()
    return ""

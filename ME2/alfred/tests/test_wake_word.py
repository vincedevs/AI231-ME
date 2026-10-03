from __future__ import annotations

from alfred.wake_word import DetectionGate


def test_detection_gate_requires_hits_and_enforces_cooldown() -> None:
    gate = DetectionGate(threshold=0.5, required_hits=2, cooldown_seconds=1.0)
    assert not gate.update(0.8, 0.0)
    assert gate.update(0.9, 0.1)
    assert not gate.update(0.9, 0.2)
    assert not gate.update(0.9, 0.3)
    assert not gate.update(0.9, 1.0)
    assert gate.update(0.9, 1.2)


def test_detection_gate_resets_consecutive_hits_on_a_miss() -> None:
    gate = DetectionGate(threshold=0.5, required_hits=2, cooldown_seconds=0)
    assert not gate.update(0.8, 0.0)
    assert not gate.update(0.1, 0.1)
    assert not gate.update(0.8, 0.2)
    assert gate.update(0.8, 0.3)

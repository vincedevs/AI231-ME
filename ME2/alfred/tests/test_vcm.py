from __future__ import annotations

import numpy as np

from alfred.vcm import (
    DEFAULT_SLOT_TOKENS,
    greedy_ctc_decode,
    greedy_ctc_decode_with_confidence,
    interpret_intent_logits,
    prepare_waveform,
)


def test_greedy_ctc_decode_collapses_repeats_and_blanks() -> None:
    token_index = {token: index for index, token in enumerate(DEFAULT_SLOT_TOKENS)}
    sequence = [token_index["t"], token_index["t"], 0, token_index["e"], token_index["n"]]
    logits = np.full((len(sequence), len(DEFAULT_SLOT_TOKENS)), -5.0, dtype=np.float32)
    for row, index in enumerate(sequence):
        logits[row, index] = 5.0
    assert greedy_ctc_decode(logits, len(sequence), DEFAULT_SLOT_TOKENS) == "ten"
    text, confidence = greedy_ctc_decode_with_confidence(logits, len(sequence), DEFAULT_SLOT_TOKENS)
    assert text == "ten"
    assert confidence > 0.99


def test_unknown_class_separates_conditional_intent_and_scope_confidence() -> None:
    logits = np.full(20, -8.0, dtype=np.float32)
    logits[4] = 2.0
    logits[19] = 3.0
    index, confidence, in_scope = interpret_intent_logits(logits, 1.0, "unknown_class")
    assert index == 4
    assert confidence > 0.99
    assert in_scope is not None
    assert in_scope < 0.5


def test_binary_scope_is_decoded_separately_from_intent_logits() -> None:
    logits = np.zeros(19, dtype=np.float32)
    logits[7] = 4.0
    index, confidence, in_scope = interpret_intent_logits(logits, 1.0, "binary_scope")
    assert index == 7
    assert confidence > 0.5
    assert in_scope is None


def test_waveform_gate_rejects_invalid_and_silent_audio_before_inference() -> None:
    assert prepare_waveform([], 400, 0.0001) is None
    assert prepare_waveform(np.zeros(16_000, dtype=np.float32), 400, 0.0001) is None
    assert prepare_waveform(np.full(16_000, 0.00001, dtype=np.float32), 400, 0.0001) is None
    assert prepare_waveform(np.asarray([np.nan], dtype=np.float32), 400, 0.0001) is None


def test_waveform_gate_preserves_valid_audio_and_pads_only_after_validation() -> None:
    audio = np.full(200, 0.01, dtype=np.float32)
    prepared = prepare_waveform(audio, 400, 0.0001)
    assert prepared is not None
    assert prepared.shape == (400,)
    np.testing.assert_array_equal(prepared[:200], audio)
    np.testing.assert_array_equal(prepared[200:], np.zeros(200, dtype=np.float32))

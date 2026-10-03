from __future__ import annotations

from alfred.speech_recognition import clean_asr_transcript


def test_asr_cleanup_removes_terminal_punctuation_names_and_repetitions() -> None:
    assert clean_asr_transcript("buy milk dot.") == "buy milk"
    assert clean_asr_transcript("for for four minutes.") == "for four minutes"


def test_asr_cleanup_does_not_remove_non_adjacent_repetition() -> None:
    assert clean_asr_transcript("very good and very clear") == "very good and very clear"

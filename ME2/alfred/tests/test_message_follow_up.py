from __future__ import annotations

from alfred.config import load_config
from alfred.message_follow_up import MessageFollowUp


def test_message_follow_up_normalizes_but_does_not_invent_content() -> None:
    follow_up = MessageFollowUp(load_config())
    payload = follow_up.compose("  dad. ", "  Please buy   milk.  ")
    assert payload is not None
    assert payload.as_mapping() == {
        "recipient": "Dad",
        "message": "Please buy milk.",
    }


def test_message_follow_up_rejects_missing_or_unsafe_content() -> None:
    follow_up = MessageFollowUp(load_config())
    assert follow_up.compose("", "Hello") is None
    assert follow_up.compose("Dad", "") is None
    assert follow_up.compose("<script>", "Hello") is None

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import AlfredConfig

RECIPIENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 '\-.]*$")


@dataclass(frozen=True)
class MessagePayload:
    recipient: str
    message: str

    def as_mapping(self) -> dict[str, str]:
        return {"recipient": self.recipient, "message": self.message}


class MessageFollowUp:
    """Validate application-level MESSAGE data without changing VCM slots."""

    def __init__(self, config: AlfredConfig) -> None:
        values = config.document["actions"]["google_chat"]
        self.recipient_prompt = str(values["recipient_prompt"])
        self.message_prompt = str(values["message_prompt"])
        self.maximum_recipient_characters = int(values["maximum_recipient_characters"])
        self.maximum_message_characters = int(values["maximum_message_characters"])

    def recipient(self, transcript: str) -> str | None:
        value = " ".join(transcript.strip().split()).strip(".,!?;:")
        if (
            not value
            or len(value) > self.maximum_recipient_characters
            or RECIPIENT_PATTERN.fullmatch(value) is None
        ):
            return None
        return value.title()

    def message(self, transcript: str) -> str | None:
        value = " ".join(transcript.strip().split())
        if not value or len(value) > self.maximum_message_characters:
            return None
        return value

    def compose(self, recipient_transcript: str, message_transcript: str) -> MessagePayload | None:
        recipient = self.recipient(recipient_transcript)
        message = self.message(message_transcript)
        if recipient is None or message is None:
            return None
        return MessagePayload(recipient=recipient, message=message)

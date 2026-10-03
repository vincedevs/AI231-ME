from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

GOOGLE_CHAT_HOST = "chat.googleapis.com"
GOOGLE_CHAT_STATUS_URL = f"https://{GOOGLE_CHAT_HOST}/"
MAXIMUM_RESPONSE_BYTES = 64 * 1024


class GoogleChatError(RuntimeError):
    """Expected Google Chat failure with a safe, speakable message."""


def validate_webhook_url(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    url = value.strip()
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != GOOGLE_CHAT_HOST
        or not parsed.path.startswith("/v1/spaces/")
        or not parsed.path.endswith("/messages")
    ):
        raise ValueError("The Google Chat webhook URL is invalid")
    parameters = urllib.parse.parse_qs(parsed.query)
    if not parameters.get("key") or not parameters.get("token"):
        raise ValueError("The Google Chat webhook URL requires key and token parameters")
    return url


class MockGoogleChatClient:
    def send_message(self, message: str, request_id: str, *, recipient: str | None = None) -> str:
        del message, request_id
        if recipient is not None:
            return f"Mock action completed. A Google Chat message to {recipient} would be sent."
        return "Mock action completed. A Google Chat message would be sent."


class GoogleChatClient:
    """Send Alfred's application-level message to one incoming webhook."""

    def __init__(
        self,
        *,
        webhook_url: str | None,
        maximum_recipient_characters: int,
        maximum_message_characters: int,
        timeout_seconds: float,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        if maximum_recipient_characters <= 0 or maximum_message_characters <= 0:
            raise ValueError("Google Chat text limits must be positive")
        if timeout_seconds <= 0:
            raise ValueError("Google Chat timeout must be positive")
        self.webhook_url = validate_webhook_url(webhook_url)
        self.maximum_recipient_characters = maximum_recipient_characters
        self.maximum_message_characters = maximum_message_characters
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def check_availability(self, timeout_seconds: float) -> bool:
        """Check Google connectivity without posting a webhook message.

        This proves that the Google Chat host is reachable, but cannot validate
        webhook credentials without creating a message.
        """
        if self.webhook_url is None:
            return False
        request = urllib.request.Request(
            GOOGLE_CHAT_STATUS_URL,
            method="HEAD",
            headers={"User-Agent": "Alfred/1.0"},
        )
        try:
            with self.opener(request, timeout=timeout_seconds):
                return True
        except urllib.error.HTTPError:
            # Any HTTP response confirms DNS, TCP, TLS, and service reachability.
            return True
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def send_message(self, message: str, request_id: str, *, recipient: str | None = None) -> str:
        message = " ".join(message.strip().split())
        if recipient is not None:
            recipient = " ".join(recipient.strip().split())
            if not recipient or len(recipient) > self.maximum_recipient_characters:
                raise GoogleChatError("The message recipient is missing or invalid.")
        if not message or len(message) > self.maximum_message_characters:
            raise GoogleChatError("The message content is missing or invalid.")
        if self.webhook_url is None:
            raise GoogleChatError("Google Chat messaging is not configured yet.")

        text = message if recipient is None else f"To: {recipient}\nMessage: {message}"
        payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.webhook_url,
            data=payload,
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "Alfred/1.0",
                "X-Request-ID": request_id,
            },
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read(MAXIMUM_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise GoogleChatError("Google Chat rejected the message.") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise GoogleChatError("Google Chat is unavailable right now.") from error
        if len(raw_response) > MAXIMUM_RESPONSE_BYTES:
            raise GoogleChatError("Google Chat returned an unexpectedly large response.")
        try:
            document = json.loads(raw_response)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GoogleChatError("Google Chat returned an invalid response.") from error
        if not isinstance(document, dict) or not str(document.get("name", "")).strip():
            raise GoogleChatError("Google Chat did not confirm that the message was sent.")
        if recipient is not None:
            return f"Your message to {recipient} has been sent."
        return "Your message has been sent."

from __future__ import annotations

import json
import urllib.error

import pytest

from alfred.google_chat import GoogleChatClient, GoogleChatError, validate_webhook_url

WEBHOOK_URL = (
    "https://chat.googleapis.com/v1/spaces/test-space/messages?key=test-key&token=test-token"
)


class Response:
    def __init__(self, document: object) -> None:
        self.payload = json.dumps(document).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self, limit: int) -> bytes:
        return self.payload[:limit]


def client(opener, webhook_url: str | None = WEBHOOK_URL) -> GoogleChatClient:
    return GoogleChatClient(
        webhook_url=webhook_url,
        maximum_recipient_characters=60,
        maximum_message_characters=500,
        timeout_seconds=5,
        opener=opener,
    )


def test_message_posts_body_without_exposing_webhook() -> None:
    captured = {}

    def opener(request, timeout):
        captured.update(timeout=timeout, body=json.loads(request.data))
        captured["content_type"] = request.get_header("Content-type")
        captured["request_id"] = request.get_header("X-request-id")
        return Response({"name": "spaces/test-space/messages/test-message"})

    result = client(opener).send_message("I will be home at six.", "request-1")

    assert result == "Your message has been sent."
    assert captured == {
        "timeout": 5,
        "body": {"text": "I will be home at six."},
        "content_type": "application/json; charset=utf-8",
        "request_id": "request-1",
    }


def test_asr_message_posts_recipient_and_message_labels() -> None:
    captured = {}

    def opener(request, timeout):
        captured.update(timeout=timeout, body=json.loads(request.data))
        return Response({"name": "spaces/test-space/messages/test-message"})

    result = client(opener).send_message(
        "I will be home at six.",
        "request-asr",
        recipient="Dad",
    )

    assert result == "Your message to Dad has been sent."
    assert captured["body"] == {"text": "To: Dad\nMessage: I will be home at six."}


@pytest.mark.parametrize(
    "url",
    [
        "http://chat.googleapis.com/v1/spaces/test/messages?key=a&token=b",
        "https://example.com/v1/spaces/test/messages?key=a&token=b",
        "https://chat.googleapis.com/v1/spaces/test/messages",
    ],
)
def test_webhook_validation_rejects_unsafe_or_incomplete_urls(url: str) -> None:
    with pytest.raises(ValueError, match="Google Chat webhook URL"):
        validate_webhook_url(url)


def test_missing_webhook_and_network_failures_become_safe_messages() -> None:
    with pytest.raises(GoogleChatError, match="not configured"):
        client(lambda request, timeout: Response({}), None).send_message("Hello.", "request-2")

    def unavailable(request, timeout):
        raise urllib.error.URLError("offline")

    with pytest.raises(GoogleChatError, match="unavailable"):
        client(unavailable).send_message("Hello.", "request-3")


def test_success_response_must_confirm_a_created_message() -> None:
    with pytest.raises(GoogleChatError, match="confirm"):
        client(lambda request, timeout: Response({})).send_message("Hello.", "request-4")


def test_availability_check_does_not_post_a_message() -> None:
    captured = []

    def opener(request, timeout):
        captured.append((request.full_url, request.method, request.data, timeout))
        return Response({})

    assert client(opener).check_availability(0.25) is True
    assert captured == [("https://chat.googleapis.com/", "HEAD", None, 0.25)]


def test_availability_check_requires_configuration_and_connectivity() -> None:
    assert client(lambda request, timeout: Response({}), None).check_availability(0.25) is False

    def unavailable(request, timeout):
        del request, timeout
        raise urllib.error.URLError("offline")

    assert client(unavailable).check_availability(0.25) is False

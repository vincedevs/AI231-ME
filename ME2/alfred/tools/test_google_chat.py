"""Validate Google Chat configuration or send one explicitly authorized test."""

from __future__ import annotations

import argparse
import sys

from alfred.actions import build_action_executor
from alfred.config import load_config, load_environment_file
from alfred.google_chat import GoogleChatClient, GoogleChatError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Alfred's Google Chat integration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Validate the private webhook without sending")
    message = subparsers.add_parser("message", help="Send one live test message")
    message.add_argument("--recipient")
    message.add_argument("--text", required=True)
    message.add_argument("--confirm-live", action="store_true")
    return parser


def build_client() -> GoogleChatClient:
    config = load_config()
    load_environment_file(config.root / ".env")
    backend = build_action_executor(config).message_backend
    if not isinstance(backend, GoogleChatClient):
        raise TypeError("Expected the real Google Chat client")
    return backend


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        client = build_client()
        if client.webhook_url is None:
            raise GoogleChatError("ALFRED_GOOGLE_CHAT_WEBHOOK_URL is missing from .env.")
        if arguments.command == "check":
            print("Google Chat webhook configuration is valid.")
            return 0
        if not arguments.confirm_live:
            print("error: add --confirm-live to authorize a live message", file=sys.stderr)
            return 2
        print(
            client.send_message(
                arguments.text,
                "manual-live-test",
                recipient=arguments.recipient,
            )
        )
        return 0
    except (GoogleChatError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

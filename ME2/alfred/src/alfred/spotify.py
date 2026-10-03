from __future__ import annotations

import logging
import uuid
from typing import Any

from .wayne_manor import WayneManorClient, WayneManorError

LOGGER = logging.getLogger(__name__)


class SpotifyError(RuntimeError):
    """Expected media-control failure that is safe to speak to the user."""


class MockSpotifyController:
    """Deterministic, in-memory music backend for development and tests."""

    def __init__(self) -> None:
        self.playback_state = "STOPPED"
        self.position_ms = 0
        self.track_number = 1
        self._resume_pending = False
        self.commands: list[str] = []

    def pause_for_interaction(self) -> None:
        self.commands.append("interaction_pause")
        self._resume_pending = self.playback_state == "PLAYING"
        if self._resume_pending:
            self.playback_state = "PAUSED"

    def resume_after_interaction(self) -> None:
        self.commands.append("interaction_complete")
        if self._resume_pending:
            self.playback_state = "PLAYING"
        self._resume_pending = False

    def play(self) -> str:
        self.commands.append("play")
        self.playback_state = "PLAYING"
        self._resume_pending = False
        return "Mock action completed. Music would start."

    def pause(self) -> str:
        self.commands.append("pause")
        if self.playback_state == "PLAYING":
            self.playback_state = "PAUSED"
        self._resume_pending = False
        return "Mock action completed. Playback would pause."

    def stop(self) -> str:
        self.commands.append("stop")
        self.playback_state = "STOPPED"
        self.position_ms = 0
        self._resume_pending = False
        return "Mock action completed. Playback would stop."

    def next(self) -> str:
        self.commands.append("next")
        self.track_number += 1
        self.position_ms = 0
        # Skipping while an interaction has temporarily paused playback must
        # preserve that resume intent, matching Wayne Manor's real controller.
        return "Mock action completed. Playback would advance."


class WayneManorSpotifyController:
    """Control Spotify in the laptop browser through Wayne Manor's local API."""

    def __init__(self, client: WayneManorClient) -> None:
        self.client = client
        self._resume_pending = False

    def _state(self) -> dict[str, Any]:
        try:
            return self.client.media_state()
        except WayneManorError as error:
            raise SpotifyError(str(error)) from error

    def _command(self, action: str) -> str:
        try:
            return self.client.media_command(action, uuid.uuid4().hex)
        except WayneManorError as error:
            raise SpotifyError(str(error)) from error

    def pause_for_interaction(self) -> None:
        self._resume_pending = False
        state = self._state()
        if state["connected"] and state["playback_state"] == "PLAYING":
            self._command("pause")
            self._resume_pending = True
            LOGGER.info("wayne_manor_spotify_interaction_paused resume_pending=true")

    def resume_after_interaction(self) -> None:
        should_resume = self._resume_pending
        self._resume_pending = False
        if should_resume:
            self._command("resume")
            LOGGER.info("wayne_manor_spotify_interaction_resumed")

    def play(self) -> str:
        self._resume_pending = False
        return self._command("play")

    def pause(self) -> str:
        already_paused = self._resume_pending
        self._resume_pending = False
        return "Music paused." if already_paused else self._command("pause")

    def stop(self) -> str:
        self._resume_pending = False
        return self._command("stop")

    def next(self) -> str:
        # Keep resume_pending intact: after advancing the paused player, the
        # state machine restores playback only if it was active before wake.
        return self._command("next")

    def adjust_volume(self, direction: int, step_percent: int) -> str:
        state = self._state()
        current = float(state.get("volume_percent", 50))
        target = max(0.0, min(100.0, current + direction * step_percent))
        try:
            return self.client.set_media_volume(target, uuid.uuid4().hex)
        except WayneManorError as error:
            raise SpotifyError(str(error)) from error

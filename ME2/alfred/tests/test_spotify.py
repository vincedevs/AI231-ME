from __future__ import annotations

import pytest

from alfred.spotify import MockSpotifyController, SpotifyError, WayneManorSpotifyController
from alfred.wayne_manor import WayneManorError


def test_mock_backend_maintains_ordered_state_without_audio() -> None:
    spotify = MockSpotifyController()
    spotify.play()
    spotify.pause_for_interaction()
    spotify.next()
    spotify.resume_after_interaction()
    assert spotify.playback_state == "PLAYING"
    assert spotify.track_number == 2
    assert spotify.commands == ["play", "interaction_pause", "next", "interaction_complete"]


class FakeWayneManor:
    def __init__(self, *, playing: bool) -> None:
        self.playback_state = "PLAYING" if playing else "STOPPED"
        self.commands: list[str] = []

    def media_state(self):
        return {
            "connected": True,
            "playback_state": self.playback_state,
            "volume_percent": 50,
        }

    def media_command(self, action, request_id):
        assert request_id
        self.commands.append(action)
        self.playback_state = {
            "play": "PLAYING",
            "resume": "PLAYING",
            "pause": "PAUSED",
            "stop": "STOPPED",
            "next": self.playback_state,
        }[action]
        return {
            "play": "Playing music through the Wayne Manor turntable.",
            "pause": "Music paused.",
            "resume": "Music resumed.",
            "stop": "Music stopped.",
            "next": "Skipping to the next track.",
        }[action]

    def set_media_volume(self, percent, request_id):
        assert request_id
        self.commands.append(f"volume:{percent:g}")
        return f"Spotify volume is now {percent:g} percent."


def test_wayne_manor_backend_pauses_and_restores_browser_playback() -> None:
    client = FakeWayneManor(playing=True)
    spotify = WayneManorSpotifyController(client)
    spotify.pause_for_interaction()
    spotify.resume_after_interaction()
    assert client.commands == ["pause", "resume"]


def test_wayne_manor_pause_intent_cancels_automatic_resume() -> None:
    client = FakeWayneManor(playing=True)
    spotify = WayneManorSpotifyController(client)
    spotify.pause_for_interaction()
    spotify.pause()
    spotify.resume_after_interaction()
    assert client.commands == ["pause"]


def test_wayne_manor_errors_are_safe_spotify_errors() -> None:
    client = FakeWayneManor(playing=False)

    def unavailable():
        raise WayneManorError("Wayne Manor is unavailable right now.")

    client.media_state = unavailable
    with pytest.raises(SpotifyError, match="unavailable right now"):
        WayneManorSpotifyController(client).pause_for_interaction()


def test_wayne_manor_volume_changes_the_browser_player() -> None:
    client = FakeWayneManor(playing=True)
    spotify = WayneManorSpotifyController(client)
    assert spotify.adjust_volume(1, 10) == "Spotify volume is now 60 percent."
    assert client.commands == ["volume:60"]

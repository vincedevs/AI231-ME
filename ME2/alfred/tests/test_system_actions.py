from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from alfred.system_actions import VolumeController, spoken_local_time


class Runner:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = iter(outputs)
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 2
        return subprocess.CompletedProcess(command, 0, next(self.outputs), "")


def which_for(*available: str):
    return lambda name: f"/usr/bin/{name}" if name in available else None


def test_time_uses_twelve_hour_spoken_format() -> None:
    assert spoken_local_time(datetime(2026, 9, 25, 0, 7, tzinfo=UTC)) == "It is 12:07 AM."
    assert spoken_local_time(datetime(2026, 9, 25, 14, 3, tzinfo=UTC)) == "It is 2:03 PM."


def test_macos_volume_is_read_then_set_and_clamped() -> None:
    runner = Runner(["98\n", ""])
    controller = VolumeController(
        step_percent=5,
        linux_maximum_percent=50,
        timeout_seconds=2,
        system="Darwin",
        which=which_for("osascript"),
        runner=runner,
    )
    assert controller.adjust(1) == 100
    assert runner.commands == [
        ["/usr/bin/osascript", "-e", "output volume of (get volume settings)"],
        ["/usr/bin/osascript", "-e", "set volume output volume 100"],
    ]


def test_linux_prefers_pipewire_and_falls_back_to_alsa() -> None:
    pipewire = Runner(["Volume: 0.42\n", ""])
    controller = VolumeController(
        step_percent=5,
        linux_maximum_percent=50,
        timeout_seconds=2,
        system="Linux",
        which=which_for("wpctl", "amixer"),
        runner=pipewire,
    )
    assert controller.adjust(-1) == 37
    assert pipewire.commands[-1] == [
        "/usr/bin/wpctl",
        "set-volume",
        "@DEFAULT_AUDIO_SINK@",
        "37%",
        "--limit",
        "0.5",
    ]

    alsa = Runner(["Mono: Playback [65%] [on]\n", ""])
    controller = VolumeController(
        step_percent=5,
        linux_maximum_percent=50,
        timeout_seconds=2,
        system="Linux",
        which=which_for("amixer"),
        runner=alsa,
    )
    assert controller.adjust(1) == 50
    assert alsa.commands[-1] == ["/usr/bin/amixer", "set", "Master", "50%"]


def test_linux_startup_clamps_existing_volume_to_fifty_percent() -> None:
    runner = Runner(["Volume: 0.78\n", ""])
    controller = VolumeController(
        step_percent=5,
        linux_maximum_percent=50,
        timeout_seconds=2,
        system="Linux",
        which=which_for("wpctl"),
        runner=runner,
    )
    assert controller.clamp_to_maximum() == 50
    assert runner.commands[-1] == [
        "/usr/bin/wpctl",
        "set-volume",
        "@DEFAULT_AUDIO_SINK@",
        "50%",
        "--limit",
        "0.5",
    ]


def test_linux_startup_does_not_raise_a_quiet_volume() -> None:
    runner = Runner(["Volume: 0.35\n"])
    controller = VolumeController(
        step_percent=5,
        linux_maximum_percent=50,
        timeout_seconds=2,
        system="Linux",
        which=which_for("wpctl"),
        runner=runner,
    )
    assert controller.clamp_to_maximum() == 35
    assert len(runner.commands) == 1

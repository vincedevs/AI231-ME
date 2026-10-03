from __future__ import annotations

import subprocess
from datetime import datetime

import pytest

from alfred.scheduling import SchedulingError, SystemdAlertScheduler, next_alarm_time


class Runner:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(command, 0, "", "")


def test_alarm_parser_selects_the_next_local_occurrence() -> None:
    now = datetime.fromisoformat("2026-09-27T20:00:00+08:00")
    assert next_alarm_time("seven thirty am", now).isoformat() == "2026-09-28T07:30:00+08:00"
    assert next_alarm_time("7 30 am", now).isoformat() == "2026-09-28T07:30:00+08:00"
    assert next_alarm_time("4 PM", now).isoformat() == "2026-09-28T16:00:00+08:00"
    assert next_alarm_time("four thirty seven PM", now).isoformat() == ("2026-09-28T16:37:00+08:00")
    assert next_alarm_time("eleven oh two am", now).isoformat() == ("2026-09-28T11:02:00+08:00")
    assert next_alarm_time("eleven zero two am", now).isoformat() == ("2026-09-28T11:02:00+08:00")
    assert next_alarm_time("at 4:45 p.m.", now).isoformat() == ("2026-09-28T16:45:00+08:00")
    assert next_alarm_time("21:15", now).isoformat() == "2026-09-27T21:15:00+08:00"
    assert next_alarm_time("noon", now).isoformat() == "2026-09-28T12:00:00+08:00"
    with pytest.raises(SchedulingError):
        next_alarm_time("sometime tomorrow", now)
    with pytest.raises(SchedulingError):
        next_alarm_time("for PM", now)


def test_systemd_scheduler_builds_bounded_argument_lists_without_a_shell() -> None:
    runner = Runner()
    now = datetime.fromisoformat("2026-09-27T20:00:00+08:00")
    scheduler = SystemdAlertScheduler(
        maximum_timer_seconds=604800,
        command_timeout_seconds=5,
        system="Linux",
        which=lambda name: f"/usr/bin/{name}" if name == "systemd-run" else None,
        runner=runner,
        clock=lambda: now,
        python_executable="/opt/alfred/.venv/bin/python",
    )
    assert scheduler.set_timer(600, "ten minutes", "timer-id") == (
        "Your timer is set for ten minutes."
    )
    assert "--on-active=600s" in runner.commands[0]
    assert runner.commands[0][-4:] == [
        "-m",
        "alfred.scheduled_alert",
        "--kind",
        "timer",
    ]
    assert scheduler.set_alarm("nine pm", "alarm-id") == ("Your alarm is set for 9:00 PM today.")
    assert "--on-calendar=2026-09-27 13:00:00 UTC" in runner.commands[1]


def test_scheduler_rejects_missing_systemd_and_excessive_duration() -> None:
    unavailable = SystemdAlertScheduler(
        maximum_timer_seconds=60,
        command_timeout_seconds=5,
        system="Darwin",
    )
    with pytest.raises(SchedulingError, match="systemd"):
        unavailable.set_timer(30, "thirty seconds", "id")
    with pytest.raises(SchedulingError, match="between"):
        unavailable.set_timer(61, "sixty one seconds", "id")

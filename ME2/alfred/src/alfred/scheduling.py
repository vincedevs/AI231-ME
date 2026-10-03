from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta


class SchedulingError(RuntimeError):
    """Expected operating-system scheduling failure with a safe message."""


HOURS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
}


def _spoken_number(value: str) -> int | None:
    """Parse an English number from zero through fifty-nine."""
    words = value.strip().split()
    if not words:
        return 0
    if len(words) == 1:
        return NUMBER_WORDS.get(words[0])
    if len(words) == 2 and words[0] in {"twenty", "thirty", "forty", "fifty"}:
        unit = NUMBER_WORDS.get(words[1])
        if unit is not None and 0 <= unit <= 9:
            return NUMBER_WORDS[words[0]] + unit
    return None


def next_alarm_time(value: str, now: datetime | None = None) -> datetime:
    """Resolve a spoken clock time to its next future local occurrence."""
    current = now or datetime.now().astimezone()
    if current.tzinfo is None:
        raise ValueError("Alarm calculations require a timezone-aware clock")
    text = " ".join(value.lower().replace(".", " ").split())
    text = text.replace("a m", "am").replace("p m", "pm")
    text = re.sub(r"^(?:at|for)\s+", "", text)

    if text == "noon":
        hour, minute = 12, 0
    elif text == "midnight":
        hour, minute = 0, 0
    else:
        numeric = re.fullmatch(r"(\d{1,2})(?:(?::|\s)(\d{2}))?\s*(am|pm)?", text)
        if numeric:
            hour = int(numeric.group(1))
            minute = int(numeric.group(2) or 0)
            meridiem = numeric.group(3)
        else:
            words = re.fullmatch(r"([a-z]+(?:\s+[a-z]+){0,2})\s+(am|pm)", text)
            if words is None:
                raise SchedulingError("I couldn't determine the alarm time.")
            clock_words = words.group(1).split()
            hour = HOURS.get(clock_words[0], -1)
            minute_words = clock_words[1:]
            if minute_words and minute_words[0] in {"o", "oh", "zero"}:
                minute_words = minute_words[1:]
            minute = _spoken_number(" ".join(minute_words))
            if hour < 0 or minute is None:
                raise SchedulingError("I couldn't determine the alarm time.")
            meridiem = words.group(2)

        if not 0 <= minute <= 59:
            raise SchedulingError("I couldn't determine the alarm time.")
        if meridiem:
            if not 1 <= hour <= 12:
                raise SchedulingError("I couldn't determine the alarm time.")
            hour = hour % 12 + (12 if meridiem == "pm" else 0)
        elif not 0 <= hour <= 23:
            raise SchedulingError("I couldn't determine the alarm time.")

    candidate = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= current:
        candidate += timedelta(days=1)
    return candidate


def spoken_alarm_time(value: datetime, now: datetime | None = None) -> str:
    current = now or datetime.now().astimezone()
    day = "today" if value.date() == current.date() else "tomorrow"
    clock = value.strftime("%I:%M %p").lstrip("0")
    return f"{clock} {day}"


class MockAlertScheduler:
    def set_timer(self, seconds: int, surface: str, request_id: str) -> str:
        del seconds, request_id
        return f"Mock action completed. A timer would be set for {surface}."

    def set_alarm(self, surface: str, request_id: str) -> str:
        del request_id
        return f"Mock action completed. An alarm would be set for {surface}."


class SystemdAlertScheduler:
    """Schedule bounded one-shot alerts through the Raspberry Pi user manager."""

    def __init__(
        self,
        *,
        maximum_timer_seconds: int,
        command_timeout_seconds: float,
        system: str | None = None,
        which: Callable[[str], str | None] = shutil.which,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        clock: Callable[[], datetime] | None = None,
        python_executable: str | None = None,
    ) -> None:
        if maximum_timer_seconds <= 0 or command_timeout_seconds <= 0:
            raise ValueError("Scheduler limits must be positive")
        self.maximum_timer_seconds = maximum_timer_seconds
        self.command_timeout_seconds = command_timeout_seconds
        self.runner = runner
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.python_executable = python_executable or sys.executable
        self.executable = (
            str(which("systemd-run")) if (system or platform.system()) == "Linux" else ""
        )

    def _schedule(self, kind: str, trigger: str, request_id: str) -> None:
        if not self.executable:
            raise SchedulingError("Timers and alarms require systemd on Raspberry Pi OS.")
        unit = f"alfred-{kind}-{request_id}"
        command = [
            self.executable,
            "--user",
            "--quiet",
            "--collect",
            f"--unit={unit}",
            trigger,
            "--timer-property=AccuracySec=1s",
            "--timer-property=Persistent=true",
            self.python_executable,
            "-m",
            "alfred.scheduled_alert",
            "--kind",
            kind,
        ]
        try:
            self.runner(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SchedulingError(f"I couldn't set the {kind} right now.") from error

    def set_timer(self, seconds: int, surface: str, request_id: str) -> str:
        if not 1 <= seconds <= self.maximum_timer_seconds:
            raise SchedulingError(
                f"Timers must be between one second and {self.maximum_timer_seconds} seconds."
            )
        self._schedule("timer", f"--on-active={seconds}s", request_id)
        return f"Your timer is set for {surface}."

    def set_alarm(self, surface: str, request_id: str) -> str:
        now = self.clock()
        due = next_alarm_time(surface, now)
        utc = due.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        self._schedule("alarm", f"--on-calendar={utc}", request_id)
        return f"Your alarm is set for {spoken_alarm_time(due, now)}."

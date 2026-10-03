from __future__ import annotations

import platform
import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime


class SystemActionError(RuntimeError):
    """Expected operating-system failure with a safe user-facing message."""


def spoken_local_time(now: datetime | None = None) -> str:
    local = now or datetime.now().astimezone()
    hour = local.strftime("%I").lstrip("0") or "12"
    return f"It is {hour}:{local.strftime('%M %p')}."


class VolumeController:
    def __init__(
        self,
        *,
        step_percent: int,
        linux_maximum_percent: int,
        timeout_seconds: float,
        system: str | None = None,
        which: Callable[[str], str | None] = shutil.which,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not 1 <= step_percent <= 100:
            raise ValueError("Volume step must be between 1 and 100 percent")
        if not 1 <= linux_maximum_percent <= 100:
            raise ValueError("Linux maximum volume must be between 1 and 100 percent")
        if timeout_seconds <= 0:
            raise ValueError("Volume command timeout must be positive")
        self.step_percent = step_percent
        self.linux_maximum_percent = linux_maximum_percent
        self.timeout_seconds = timeout_seconds
        self.runner = runner
        operating_system = system or platform.system()
        if operating_system == "Darwin" and which("osascript"):
            self.backend = "macos"
            self.executable = str(which("osascript"))
        elif operating_system == "Linux" and which("wpctl"):
            self.backend = "pipewire"
            self.executable = str(which("wpctl"))
        elif operating_system == "Linux" and which("amixer"):
            self.backend = "alsa"
            self.executable = str(which("amixer"))
        else:
            self.backend = None
            self.executable = None

    @property
    def maximum_percent(self) -> int:
        """Return Alfred's ceiling without limiting the macOS development host."""
        return self.linux_maximum_percent if self.backend in {"pipewire", "alsa"} else 100

    def _run(self, arguments: list[str]) -> str:
        if self.executable is None:
            raise SystemActionError("System volume control is unavailable.")
        try:
            completed = self.runner(
                [self.executable, *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SystemActionError("System volume control is unavailable.") from error
        return completed.stdout.strip()

    def current_percent(self) -> int:
        if self.backend == "macos":
            output = self._run(["-e", "output volume of (get volume settings)"])
            match = re.search(r"\d+", output)
        elif self.backend == "pipewire":
            output = self._run(["get-volume", "@DEFAULT_AUDIO_SINK@"])
            match = re.search(r"Volume:\s*([0-9.]+)", output)
            if match:
                return max(0, min(100, round(float(match.group(1)) * 100)))
        elif self.backend == "alsa":
            output = self._run(["get", "Master"])
            matches = re.findall(r"\[(\d+)%\]", output)
            match = re.match(r"\d+", matches[-1]) if matches else None
        else:
            raise SystemActionError("System volume control is unavailable.")
        if match is None:
            raise SystemActionError("I couldn't determine the current system volume.")
        return max(0, min(100, int(match.group(0))))

    def _set_percent(self, target: int) -> None:
        if self.backend == "macos":
            self._run(["-e", f"set volume output volume {target}"])
        elif self.backend == "pipewire":
            self._run(
                [
                    "set-volume",
                    "@DEFAULT_AUDIO_SINK@",
                    f"{target}%",
                    "--limit",
                    f"{self.maximum_percent / 100:g}",
                ]
            )
        elif self.backend == "alsa":
            self._run(["set", "Master", f"{target}%"])
        else:
            raise SystemActionError("System volume control is unavailable.")

    def adjust(self, direction: int) -> int:
        if direction not in {-1, 1}:
            raise ValueError("Volume direction must be -1 or 1")
        current = self.current_percent()
        target = max(0, min(self.maximum_percent, current + direction * self.step_percent))
        self._set_percent(target)
        return target

    def clamp_to_maximum(self) -> int:
        """Lower Linux volume at startup when it exceeds Alfred's configured ceiling."""
        current = self.current_percent()
        target = min(current, self.maximum_percent)
        if target != current:
            self._set_percent(target)
        return target

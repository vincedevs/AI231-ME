from __future__ import annotations

import subprocess
from collections.abc import Callable
from typing import Any


class LinphoneCallError(RuntimeError):
    """Expected Linphone failure with a safe user-facing message."""


class MockCallController:
    """No-side-effect CALL backend used by Alfred's explicit mock mode."""

    def call(self, request_id: str) -> str:
        del request_id
        return "Mock action completed. A call would start."


class LinphoneCallController:
    """Start one configured SIP call through the local ``linphonecsh`` daemon."""

    def __init__(
        self,
        *,
        destination: str,
        executable: str = "linphonecsh",
        timeout_seconds: float = 8.0,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        normalized_destination = destination.strip()
        if not normalized_destination.startswith(("sip:", "sips:")):
            raise ValueError("The Linphone destination must be a sip: or sips: URI")
        if len(normalized_destination) > 512 or any(
            character.isspace() or not character.isprintable()
            for character in normalized_destination
        ):
            raise ValueError("The Linphone destination is invalid")
        if not executable.strip():
            raise ValueError("The Linphone executable cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("The Linphone command timeout must be positive")
        self.destination = normalized_destination
        self.executable = executable.strip()
        self.timeout_seconds = timeout_seconds
        self.runner = runner

    def _run(self, arguments: list[str]) -> Any:
        try:
            return self.runner(
                arguments,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as error:
            raise LinphoneCallError("Linphone is not installed on this device.") from error
        except subprocess.TimeoutExpired as error:
            raise LinphoneCallError("Linphone did not respond in time.") from error
        except OSError as error:
            raise LinphoneCallError("Linphone is unavailable right now.") from error

    @staticmethod
    def _failed(completed: Any) -> bool:
        output = f"{completed.stdout or ''}\n{completed.stderr or ''}".casefold()
        return completed.returncode != 0 or "error" in output or "failed" in output

    def call(self, request_id: str) -> str:
        del request_id
        initialized = self._run([self.executable, "init"])
        if self._failed(initialized):
            raise LinphoneCallError(
                "Linphone could not start. Check its account and daemon configuration."
            )
        dialed = self._run([self.executable, "dial", self.destination])
        if self._failed(dialed):
            raise LinphoneCallError("Linphone could not place the call.")
        return "Linphone has started the call."

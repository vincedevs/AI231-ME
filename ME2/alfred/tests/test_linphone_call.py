from __future__ import annotations

import subprocess

import pytest

from alfred.linphone_call import LinphoneCallController, LinphoneCallError


class Completed:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_linphone_initializes_daemon_then_dials_configured_sip_uri() -> None:
    calls = []

    def runner(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return Completed(stdout="Establishing call id 2")

    controller = LinphoneCallController(
        destination="sip:dad@sip.linphone.org",
        timeout_seconds=4,
        runner=runner,
    )

    assert controller.call("request-1") == "Linphone has started the call."
    assert [arguments for arguments, _ in calls] == [
        ["linphonecsh", "init"],
        ["linphonecsh", "dial", "sip:dad@sip.linphone.org"],
    ]
    assert calls[0][1] == {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": 4,
    }


def test_linphone_treats_reported_invite_error_as_failure() -> None:
    responses = iter(
        [
            Completed(),
            Completed(stdout="Call 2 error.\nError from linphone_core_invite."),
        ]
    )
    controller = LinphoneCallController(
        destination="sip:dad@sip.linphone.org",
        runner=lambda *args, **kwargs: next(responses),
    )

    with pytest.raises(LinphoneCallError, match="could not place"):
        controller.call("request-2")


def test_linphone_reports_missing_binary_and_rejects_non_sip_destination() -> None:
    with pytest.raises(ValueError, match="sip:"):
        LinphoneCallController(destination="+639171234567")

    def missing(*args, **kwargs):
        raise FileNotFoundError

    controller = LinphoneCallController(
        destination="sip:dad@sip.linphone.org",
        runner=missing,
    )
    with pytest.raises(LinphoneCallError, match="not installed"):
        controller.call("request-3")


def test_linphone_timeout_has_safe_error() -> None:
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("linphonecsh", 8)

    controller = LinphoneCallController(
        destination="sip:dad@sip.linphone.org",
        runner=timeout,
    )
    with pytest.raises(LinphoneCallError, match="did not respond"):
        controller.call("request-4")

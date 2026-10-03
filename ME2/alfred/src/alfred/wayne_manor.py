from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

MAX_RESPONSE_BYTES = 256_000
MAX_REQUEST_BYTES = 16_384


class WayneManorError(RuntimeError):
    """Expected WayneManor API failure with a safe user-facing message."""


class WayneManorClient:
    """Small synchronous client for Alfred's local WayneManor API."""

    def __init__(
        self,
        *,
        base_url: str,
        light_group: str,
        thermostat: str,
        temperature_unit: str,
        timeout_seconds: float,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Wayne Manor base URL must be an HTTP or HTTPS URL")
        if not light_group.strip() or not thermostat.strip():
            raise ValueError("Wayne Manor device identifiers cannot be empty")
        if temperature_unit not in {"celsius", "fahrenheit"}:
            raise ValueError("Wayne Manor temperature unit must be celsius or fahrenheit")
        if timeout_seconds <= 0:
            raise ValueError("Wayne Manor timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.light_group = light_group.strip()
        self.thermostat = thermostat.strip()
        self.temperature_unit = temperature_unit
        self.timeout_seconds = timeout_seconds
        self.opener = opener

    def _request(self, method: str, path: str, body: dict[str, Any] | None, request_id: str) -> str:
        if method not in {"POST", "PUT"}:
            raise ValueError("Wayne Manor mutations must use POST or PUT")
        body = {} if body is None else body
        payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_REQUEST_BYTES:
            raise WayneManorError("The WayneManor API command is unexpectedly large.")
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=payload,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "Alfred/1.0",
                "X-Request-ID": request_id,
            },
        )
        try:
            with self.opener(request, timeout=self.timeout_seconds) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            response_body = error.read(MAX_RESPONSE_BYTES + 1)
            message = self._error_message(response_body)
            raise WayneManorError(message or "WayneManor API rejected the command.") from error
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            raise WayneManorError("Wayne Manor is unavailable right now.") from error
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise WayneManorError("WayneManor API returned an unexpectedly large response.")
        try:
            document = json.loads(response_body)
            message = document["message"]
            echoed_request_id = document["request_id"]
            if document.get("status") != "success" or not isinstance(message, str):
                raise ValueError
            if echoed_request_id != request_id:
                raise ValueError
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise WayneManorError("WayneManor API returned an invalid response.") from error
        return message

    def _put(self, path: str, body: dict[str, Any], request_id: str) -> str:
        return self._request("PUT", path, body, request_id)

    def _get(self, path: str, *, timeout_seconds: float | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            method="GET",
            headers={"Accept": "application/json", "User-Agent": "Alfred/1.0"},
        )
        try:
            with self.opener(
                request,
                timeout=self.timeout_seconds if timeout_seconds is None else timeout_seconds,
            ) as response:
                response_body = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            message = self._error_message(error.read(MAX_RESPONSE_BYTES + 1))
            raise WayneManorError(message or "WayneManor API rejected the request.") from error
        except (OSError, TimeoutError, urllib.error.URLError) as error:
            raise WayneManorError("Wayne Manor is unavailable right now.") from error
        if len(response_body) > MAX_RESPONSE_BYTES:
            raise WayneManorError("WayneManor API returned an unexpectedly large response.")
        try:
            document = json.loads(response_body)
        except (TypeError, json.JSONDecodeError) as error:
            raise WayneManorError("WayneManor API returned an invalid response.") from error
        if not isinstance(document, dict):
            raise WayneManorError("WayneManor API returned an invalid response.")
        return document

    def check_availability(self, timeout_seconds: float) -> bool:
        """Probe Wayne Manor without mutating any simulated device state."""
        try:
            document = self._get("health", timeout_seconds=timeout_seconds)
        except WayneManorError:
            return False
        return document.get("status") == "ok"

    @staticmethod
    def _error_message(payload: bytes) -> str | None:
        if len(payload) > MAX_RESPONSE_BYTES:
            return None
        try:
            message = json.loads(payload)["error"]["message"]
        except (KeyError, TypeError, json.JSONDecodeError):
            return None
        return message if isinstance(message, str) and message.strip() else None

    @staticmethod
    def _device_path(kind: str, device_id: str, action: str) -> str:
        safe_id = urllib.parse.quote(device_id, safe="")
        return f"{kind}/{safe_id}/{action}"

    def set_light_power(self, on: bool, request_id: str) -> str:
        path = self._device_path("light-groups", self.light_group, "power")
        return self._put(path, {"on": on}, request_id)

    def set_brightness(self, percent: float, request_id: str) -> str:
        path = self._device_path("light-groups", self.light_group, "brightness")
        return self._put(path, {"percent": percent}, request_id)

    def set_color(self, color: str, request_id: str) -> str:
        path = self._device_path("light-groups", self.light_group, "color")
        return self._put(path, {"color": color}, request_id)

    def set_temperature(self, degrees: float, request_id: str) -> str:
        path = self._device_path("thermostats", self.thermostat, "setpoint")
        return self._put(
            path,
            {"degrees": degrees, "unit": self.temperature_unit},
            request_id,
        )

    def call(self, request_id: str) -> str:
        """Ring the simulated Wayne Manor telephone for a local CALL demo."""
        return self._request("POST", "telephone/calls", {}, request_id)

    def media_state(self) -> dict[str, Any]:
        state = self._get("media")
        if not isinstance(state.get("connected"), bool) or state.get("playback_state") not in {
            "STOPPED",
            "PLAYING",
            "PAUSED",
        }:
            raise WayneManorError("WayneManor API returned an invalid media state.")
        return state

    def media_command(self, action: str, request_id: str) -> str:
        if action not in {"play", "pause", "resume", "stop", "next"}:
            raise ValueError(f"Unsupported Wayne Manor media action: {action}")
        return self._request("POST", f"media/{action}", {}, request_id)

    def set_media_volume(self, percent: float, request_id: str) -> str:
        return self._request("PUT", "media/volume", {"percent": percent}, request_id)

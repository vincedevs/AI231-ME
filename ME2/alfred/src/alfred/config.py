from __future__ import annotations

import hashlib
import json
import os
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

APPLICATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = APPLICATION_ROOT / "config" / "default.json"
APPROVED_REAL_HANDLERS = {
    "play_music",
    "pause_music",
    "stop_music",
    "next_music",
    "get_weather",
    "get_time",
    "send_message",
    "increase_volume",
    "decrease_volume",
    "create_reminder",
    "list_reminders",
    "turn_on_wayne_manor_lights",
    "turn_off_wayne_manor_lights",
    "set_wayne_manor_temperature",
    "set_wayne_manor_brightness",
    "set_wayne_manor_color",
    "start_call",
    "set_timer",
    "set_alarm",
}
REAL_HANDLER_BY_INTENT = {
    "PLAY_MUSIC": "play_music",
    "PAUSE": "pause_music",
    "STOP": "stop_music",
    "NEXT": "next_music",
    "WEATHER": "get_weather",
    "TIME": "get_time",
    "MESSAGE": "send_message",
    "VOLUME_UP": "increase_volume",
    "VOLUME_DOWN": "decrease_volume",
    "CREATE_REMINDER": "create_reminder",
    "LIST_REMINDERS": "list_reminders",
    "LIGHT_ON": "turn_on_wayne_manor_lights",
    "LIGHT_OFF": "turn_off_wayne_manor_lights",
    "TEMPERATURE": "set_wayne_manor_temperature",
    "BRIGHTNESS": "set_wayne_manor_brightness",
    "COLOR": "set_wayne_manor_color",
    "CALL": "start_call",
    "TIMER": "set_timer",
    "ALARM": "set_alarm",
}


def load_environment_file(path: Path) -> bool:
    """Load a small dotenv file without executing it as shell code.

    Existing process variables take precedence, which preserves explicit
    deployment overrides. Values may be unquoted or wrapped in matching single
    or double quotes; interpolation and shell syntax are deliberately unsupported.
    """
    if not path.is_file():
        return False
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid environment entry at {path}:{line_number}")
        name, value = (item.strip() for item in line.split("=", 1))
        if not name.isidentifier():
            raise ValueError(f"Invalid environment name at {path}:{line_number}")
        if value[:1] in {"'", '"'}:
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"Unclosed environment quote at {path}:{line_number}")
            value = value[1:-1]
        os.environ.setdefault(name, value)
    return True


def update_environment_file(path: Path, name: str, value: str) -> None:
    """Persist one literal dotenv value while preserving all other entries."""
    if not name.isidentifier():
        raise ValueError("Environment name must be a valid identifier")
    if not value or any(character in value for character in "\r\n\"'"):
        raise ValueError("Environment value contains unsupported characters")
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    replacement = f'{name}="{value}"'
    updated: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith(f"{name}="):
            if not replaced:
                updated.append(replacement)
                replaced = True
            continue
        updated.append(line)
    if not replaced:
        updated.append(replacement)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(updated) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required Alfred file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_directory(path: Path) -> tuple[str, int]:
    """Hash a model directory using its relative paths and file contents."""
    if not path.is_dir():
        raise FileNotFoundError(f"Required directory is missing: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest(), len(files)


@dataclass(frozen=True)
class AlfredConfig:
    root: Path
    document: dict[str, Any]
    schema: dict[str, Any]
    action_contract: dict[str, Any]
    feedback: dict[str, Any]
    model_manifest: dict[str, Any]

    def local_path(self, value: str) -> Path:
        candidate = (self.root / value).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"Alfred path escapes its application folder: {value}") from error
        return candidate

    @property
    def intents(self) -> tuple[str, ...]:
        return tuple(row["name"] for row in self.schema["intents"])

    @property
    def slot_by_intent(self) -> dict[str, str]:
        return {
            row["name"]: row["slot"] for row in self.schema["intents"] if row["slot"] is not None
        }

    @property
    def action_by_intent(self) -> dict[str, dict[str, Any]]:
        return {row["intent"]: row for row in self.action_contract["actions"]}


def _load_relative(root: Path, paths: dict[str, Any], name: str) -> dict[str, Any]:
    value = paths.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"paths.{name} must be a non-empty relative path")
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"paths.{name} escapes the Alfred folder") from error
    return read_json(candidate)


def validate_config(config: AlfredConfig) -> None:
    audio = config.document.get("audio", {})
    if audio.get("sample_rate") != 16_000:
        raise ValueError("Alfred currently requires 16 kHz audio")
    if int(audio.get("channels", 0)) != 1:
        raise ValueError("Alfred currently requires mono audio")
    if int(audio.get("block_samples", 0)) <= 0:
        raise ValueError("audio.block_samples must be positive")
    if int(audio.get("block_samples", 0)) != 480:
        raise ValueError("The bundled recurrent Silero VAD requires 480-sample frames")
    for name in ("feedback_gain", "tts_gain"):
        if not 0 < float(audio.get(name, 0)) <= 1:
            raise ValueError(f"audio.{name} must be greater than 0 and at most 1")
    if not isinstance(audio.get("use_system_default_output"), bool):
        raise TypeError("audio.use_system_default_output must be true or false")

    wake_word = config.document.get("wake_word", {})
    if int(wake_word.get("chunk_samples", 0)) != 1_280:
        raise ValueError("The bundled wake-word frontend requires 1,280-sample chunks")
    for name in ("hey_alfred_threshold", "im_batman_threshold"):
        if not 0 <= float(wake_word.get(name, -1)) <= 1:
            raise ValueError(f"wake_word.{name} must be between 0 and 1")
    if int(wake_word.get("required_hits", 0)) <= 0:
        raise ValueError("wake_word.required_hits must be positive")

    names = config.intents
    if len(names) != 19 or len(set(names)) != 19:
        raise ValueError("The VCM schema must contain exactly 19 unique intents")
    action_names = tuple(row["intent"] for row in config.action_contract.get("actions", []))
    if set(action_names) != set(names) or len(action_names) != len(names):
        raise ValueError("The action contract must map every canonical intent exactly once")
    if config.action_contract.get("mode") != "mixed":
        raise ValueError("The action contract must use the reviewed mixed mode")
    rules = config.action_contract.get("rules", {})
    if rules.get("real_external_side_effects_enabled") is not True:
        raise ValueError("The mixed action contract must explicitly enable reviewed side effects")
    approved = set(rules.get("approved_real_handlers", []))
    if approved != APPROVED_REAL_HANDLERS:
        raise ValueError("The action contract's approved real handlers have changed")
    handlers = tuple(str(row.get("handler", "")) for row in config.action_contract["actions"])
    if len(set(handlers)) != 19:
        raise ValueError("Every canonical intent must use one unique handler")
    if any(name not in approved and not name.startswith("mock_") for name in handlers):
        raise ValueError("The action contract contains an unapproved real handler")
    used_real_handlers = {name for name in handlers if not name.startswith("mock_")}
    if used_real_handlers != approved:
        raise ValueError("Every approved real handler must be mapped exactly once")
    for intent, handler in REAL_HANDLER_BY_INTENT.items():
        if config.action_by_intent[intent].get("handler") != handler:
            raise ValueError(f"The reviewed real handler for {intent} must be {handler}")
    for intent in names:
        expected = [config.slot_by_intent[intent]] if intent in config.slot_by_intent else []
        if config.action_by_intent[intent].get("required_slots") != expected:
            raise ValueError(f"Required action slots do not match the schema for {intent}")

    vcm = config.document.get("vcm", {})
    if vcm.get("mode") not in {"mock", "onnx"}:
        raise ValueError("vcm.mode must be 'mock' or 'onnx'")
    threshold = float(vcm.get("minimum_intent_confidence", -1))
    if not 0 <= threshold <= 1:
        raise ValueError("vcm.minimum_intent_confidence must be between 0 and 1")
    minimum_waveform_rms = float(vcm.get("minimum_waveform_rms", -1))
    if not 0 < minimum_waveform_rms <= 1:
        raise ValueError("vcm.minimum_waveform_rms must be greater than 0 and at most 1")
    if int(vcm.get("onnx_threads", 0)) <= 0:
        raise ValueError("vcm.onnx_threads must be positive")

    tts = config.document.get("tts", {})
    if not isinstance(tts.get("enabled"), bool):
        raise TypeError("tts.enabled must be true or false")
    if tts.get("engine") != "sherpa_onnx_vits":
        raise ValueError("Alfred currently supports only the sherpa_onnx_vits TTS engine")
    if int(tts.get("threads", 0)) <= 0:
        raise ValueError("tts.threads must be positive")
    if float(tts.get("speed", 0)) <= 0:
        raise ValueError("tts.speed must be positive")
    if float(tts.get("silence_scale", -1)) < 0:
        raise ValueError("tts.silence_scale cannot be negative")
    if not 0 <= float(tts.get("response_segment_pause_seconds", -1)) <= 2:
        raise ValueError("tts.response_segment_pause_seconds must be between 0 and 2")
    packaged_tts = config.model_manifest.get("artifacts", {}).get("tts")
    if tts["enabled"] and not isinstance(packaged_tts, dict):
        raise ValueError("Enabled TTS requires a packaged tts artifact")
    if tts["enabled"] and packaged_tts.get("engine") != tts.get("engine"):
        raise ValueError("Configured TTS engine does not match the packaged artifact")
    if tts["enabled"] and packaged_tts.get("voice") != tts.get("voice"):
        raise ValueError("Configured TTS voice does not match the packaged artifact")

    actions = config.document.get("actions", {})
    enabled_intents = actions.get("enabled_intents")
    if not isinstance(enabled_intents, list) or not enabled_intents:
        raise ValueError("actions.enabled_intents must be a non-empty list")
    if any(not isinstance(intent, str) for intent in enabled_intents):
        raise TypeError("actions.enabled_intents must contain only intent names")
    if len(enabled_intents) != len(set(enabled_intents)):
        raise ValueError("actions.enabled_intents cannot contain duplicates")
    unknown_enabled = set(enabled_intents) - set(names)
    if unknown_enabled:
        raise ValueError(
            f"actions.enabled_intents contains non-canonical intents: {sorted(unknown_enabled)}"
        )
    database_path = actions.get("database_path")
    if not isinstance(database_path, str) or not database_path:
        raise ValueError("actions.database_path must be a non-empty relative path")
    config.local_path(database_path)
    weather = actions.get("weather", {})
    if not str(weather.get("location_name", "")).strip():
        raise ValueError("actions.weather.location_name cannot be empty")
    if not -90 <= float(weather.get("latitude", 1000)) <= 90:
        raise ValueError("actions.weather.latitude must be between -90 and 90")
    if not -180 <= float(weather.get("longitude", 1000)) <= 180:
        raise ValueError("actions.weather.longitude must be between -180 and 180")
    if weather.get("temperature_unit") not in {"celsius", "fahrenheit"}:
        raise ValueError("actions.weather.temperature_unit must be celsius or fahrenheit")
    if not str(weather.get("location_prompt", "")).strip():
        raise ValueError("actions.weather.location_prompt cannot be empty")
    if not 2 <= int(weather.get("maximum_location_characters", 0)) <= 200:
        raise ValueError("actions.weather.maximum_location_characters must be 2–200")
    if float(weather.get("timeout_seconds", 0)) <= 0:
        raise ValueError("actions.weather.timeout_seconds must be positive")
    if float(weather.get("cache_seconds", -1)) < 0:
        raise ValueError("actions.weather.cache_seconds cannot be negative")
    volume = actions.get("volume", {})
    if not 1 <= int(volume.get("step_percent", 0)) <= 100:
        raise ValueError("actions.volume.step_percent must be between 1 and 100")
    if not 1 <= int(volume.get("linux_maximum_percent", 0)) <= 100:
        raise ValueError("actions.volume.linux_maximum_percent must be between 1 and 100")
    if float(volume.get("timeout_seconds", 0)) <= 0:
        raise ValueError("actions.volume.timeout_seconds must be positive")
    scheduler = actions.get("scheduler", {})
    if int(scheduler.get("maximum_timer_seconds", 0)) <= 0:
        raise ValueError("actions.scheduler.maximum_timer_seconds must be positive")
    if float(scheduler.get("command_timeout_seconds", 0)) <= 0:
        raise ValueError("actions.scheduler.command_timeout_seconds must be positive")
    if int(scheduler.get("alert_repeat_count", 0)) <= 0:
        raise ValueError("actions.scheduler.alert_repeat_count must be positive")
    if float(scheduler.get("alert_interval_seconds", -1)) < 0:
        raise ValueError("actions.scheduler.alert_interval_seconds cannot be negative")
    reminders = actions.get("reminders", {})
    if int(reminders.get("max_task_characters", 0)) <= 0:
        raise ValueError("actions.reminders.max_task_characters must be positive")
    spoken_limit = reminders.get("spoken_limit")
    if spoken_limit is not None and int(spoken_limit) <= 0:
        raise ValueError("actions.reminders.spoken_limit must be null or positive")
    call = actions.get("call", {})
    if call.get("backend") not in {"linphone", "wayne_manor"}:
        raise ValueError("actions.call.backend must be linphone or wayne_manor")
    if not str(call.get("linphone_executable", "")).strip():
        raise ValueError("actions.call.linphone_executable cannot be empty")
    destination_environment = str(call.get("linphone_destination_environment", ""))
    if not destination_environment.isidentifier():
        raise ValueError(
            "actions.call.linphone_destination_environment must be an environment name"
        )
    if float(call.get("command_timeout_seconds", 0)) <= 0:
        raise ValueError("actions.call.command_timeout_seconds must be positive")
    google_chat = actions.get("google_chat", {})
    if not str(google_chat.get("webhook_url_environment", "")).isidentifier():
        raise ValueError("actions.google_chat.webhook_url_environment must be an environment name")
    for name in ("recipient_prompt", "message_prompt"):
        if not str(google_chat.get(name, "")).strip():
            raise ValueError(f"actions.google_chat.{name} cannot be empty")
    if not 1 <= int(google_chat.get("maximum_recipient_characters", 0)) <= 200:
        raise ValueError("actions.google_chat.maximum_recipient_characters must be 1–200")
    if not 1 <= int(google_chat.get("maximum_message_characters", 0)) <= 4_000:
        raise ValueError("actions.google_chat.maximum_message_characters must be 1–4000")
    fallback_messages = google_chat.get("fallback_messages")
    if not isinstance(fallback_messages, list) or not 1 <= len(fallback_messages) <= 5:
        raise ValueError("actions.google_chat.fallback_messages requires one to five messages")
    if any(not isinstance(item, str) or not item.strip() for item in fallback_messages):
        raise ValueError("actions.google_chat.fallback_messages cannot contain empty values")
    if len(set(fallback_messages)) != len(fallback_messages):
        raise ValueError("actions.google_chat.fallback_messages must be unique")
    if any(
        len(item) > int(google_chat["maximum_message_characters"]) for item in fallback_messages
    ):
        raise ValueError("actions.google_chat.fallback_messages contains an oversized message")
    fallback_message_seed = google_chat.get("fallback_message_seed")
    if fallback_message_seed is not None and not isinstance(fallback_message_seed, int):
        raise ValueError("actions.google_chat.fallback_message_seed must be null or an integer")
    if float(google_chat.get("timeout_seconds", 0)) <= 0:
        raise ValueError("actions.google_chat.timeout_seconds must be positive")
    wayne_manor = actions.get("wayne_manor", {})
    wayne_manor_url = urllib.parse.urlparse(str(wayne_manor.get("base_url", "")))
    if wayne_manor_url.scheme not in {"http", "https"} or not wayne_manor_url.netloc:
        raise ValueError("actions.wayne_manor.base_url must be an HTTP or HTTPS URL")
    wayne_manor_url_environment = str(wayne_manor.get("base_url_environment", ""))
    if not wayne_manor_url_environment.isidentifier():
        raise ValueError("actions.wayne_manor.base_url_environment must be an environment name")
    if not str(wayne_manor.get("default_light_group", "")).strip():
        raise ValueError("actions.wayne_manor.default_light_group cannot be empty")
    if not str(wayne_manor.get("default_thermostat", "")).strip():
        raise ValueError("actions.wayne_manor.default_thermostat cannot be empty")
    if wayne_manor.get("temperature_unit") not in {"celsius", "fahrenheit"}:
        raise ValueError("actions.wayne_manor.temperature_unit must be celsius or fahrenheit")
    if float(wayne_manor.get("timeout_seconds", 0)) <= 0:
        raise ValueError("actions.wayne_manor.timeout_seconds must be positive")
    fallback_temperature = float(wayne_manor.get("fallback_temperature_degrees", -1))
    minimum_temperature = float(wayne_manor.get("minimum_temperature_degrees", 0))
    maximum_temperature = float(wayne_manor.get("maximum_temperature_degrees", 0))
    if not minimum_temperature <= fallback_temperature <= maximum_temperature:
        raise ValueError("The fallback temperature must be within the configured safety range")
    if not 0 <= float(wayne_manor.get("fallback_brightness_percent", -1)) <= 100:
        raise ValueError("actions.wayne_manor.fallback_brightness_percent must be 0–100")
    fallback_colors = wayne_manor.get("fallback_colors")
    if not isinstance(fallback_colors, list) or len(fallback_colors) < 2:
        raise ValueError("actions.wayne_manor.fallback_colors requires at least two colors")
    if any(not isinstance(item, str) or not item.strip() for item in fallback_colors):
        raise ValueError("actions.wayne_manor.fallback_colors cannot contain empty values")
    if len(set(fallback_colors)) != len(fallback_colors):
        raise ValueError("actions.wayne_manor.fallback_colors must be unique")
    if not isinstance(wayne_manor.get("fallback_color_seed"), int):
        raise TypeError("actions.wayne_manor.fallback_color_seed must be an integer")

    runtime = config.document.get("runtime", {})
    if not 0 < float(runtime.get("api_probe_timeout_seconds", 0)) <= 10:
        raise ValueError("runtime.api_probe_timeout_seconds must be greater than 0 and at most 10")

    capture = config.document.get("command_capture", {})
    if not 0 <= float(capture.get("vad_threshold", -1)) <= 1:
        raise ValueError("command_capture.vad_threshold must be between 0 and 1")
    positive_capture_values = (
        "speech_start_ms",
        "pre_roll_ms",
        "minimum_speech_ms",
        "end_silence_ms",
        "speech_start_timeout_seconds",
        "maximum_command_seconds",
    )
    if any(float(capture.get(name, 0)) <= 0 for name in positive_capture_values):
        raise ValueError("Command capture timing values must be positive")
    if float(capture.get("post_roll_ms", -1)) < 0:
        raise ValueError("command_capture.post_roll_ms cannot be negative")

    clarification = config.document.get("clarification", {})
    if not isinstance(clarification.get("enabled"), bool):
        raise TypeError("clarification.enabled must be true or false")
    if not 0 <= float(clarification.get("minimum_slot_confidence", -1)) <= 1:
        raise ValueError("clarification.minimum_slot_confidence must be between 0 and 1")
    if float(clarification.get("maximum_response_seconds", 0)) <= 0:
        raise ValueError("clarification.maximum_response_seconds must be positive")
    expected_clarification_intents = set(config.slot_by_intent)
    prompts = clarification.get("prompts")
    if not isinstance(prompts, dict) or set(prompts) != expected_clarification_intents:
        raise ValueError("Clarification prompts must exactly match the six slot-bearing intents")
    if any(not isinstance(value, str) or not value.strip() for value in prompts.values()):
        raise ValueError("Clarification prompts cannot be empty")
    asr = clarification.get("asr", {})
    if asr.get("engine") != "sherpa_onnx_moonshine_v2":
        raise ValueError("Alfred clarification supports only Moonshine v2")
    for name in ("encoder_path", "decoder_path", "tokens_path"):
        value = asr.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"clarification.asr.{name} must be a relative path")
        config.local_path(value)
    if int(asr.get("threads", 0)) <= 0:
        raise ValueError("clarification.asr.threads must be positive")

    outcomes = config.schema.get("rejection_policy", {}).get("outcomes", [])
    if outcomes != ["execute", "low_confidence", "unsupported"]:
        raise ValueError("The schema must preserve the three canonical VCM decisions")

    announcements = config.feedback.get("announcements", {})
    if set(announcements) != set(names):
        raise ValueError("Feedback must define one announcement for every canonical intent")
    for intent, template in announcements.items():
        placeholders = {
            field_name
            for _, field_name, _, _ in Formatter().parse(str(template))
            if field_name is not None
        }
        allowed = {config.slot_by_intent[intent]} if intent in config.slot_by_intent else set()
        if placeholders != allowed:
            raise ValueError(f"Announcement placeholders do not match the schema for {intent}")


def load_config(path: Path | None = None) -> AlfredConfig:
    config_path = (path or DEFAULT_CONFIG).expanduser().resolve()
    root = APPLICATION_ROOT
    document = read_json(config_path)
    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise TypeError("The Alfred configuration requires a paths object")
    model_manifest_path = (root / str(paths.get("model_manifest", ""))).resolve()
    model_manifest = (
        read_json(model_manifest_path)
        if model_manifest_path.is_file()
        else {
            "bundle_version": "unpackaged",
            "artifacts": {},
        }
    )
    config = AlfredConfig(
        root=root,
        document=document,
        schema=_load_relative(root, paths, "vcm_schema"),
        action_contract=_load_relative(root, paths, "action_contract"),
        feedback=_load_relative(root, paths, "feedback"),
        model_manifest=model_manifest,
    )
    validate_config(config)
    return config

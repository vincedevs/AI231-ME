from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import urllib.parse
from collections.abc import Callable
from pathlib import Path

from .actions import build_action_executor
from .audio import MicrophoneInput, matching_device_indices, playback_output_device
from .benchmark_logging import BenchmarkEventLogger
from .clarification import build_slot_clarifier
from .config import (
    DEFAULT_CONFIG,
    AlfredConfig,
    load_config,
    load_environment_file,
    update_environment_file,
)
from .health import verify_install
from .message_follow_up import MessageFollowUp
from .playback import FeedbackPlayer, NullFeedbackPlayer
from .reminders import initialize_database, reset_database
from .speech_recognition import build_speech_recognizer
from .state_machine import AlfredApplication
from .system_actions import SystemActionError
from .tts import build_tts
from .vad import CommandCapture, SileroVad
from .vcm import build_vcm
from .wake_word import DualWakeWordDetector

LOGGER = logging.getLogger(__name__)
APPLICATION_SERVICES = ("alfred.service",)


def parse_slot(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("mock slots must use NAME=VALUE")
    name, item = value.split("=", 1)
    if not name.strip() or not item.strip():
        raise argparse.ArgumentTypeError("mock slots require a non-empty name and value")
    return name.strip(), item.strip()


def environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def weather_asr_enabled(args: argparse.Namespace, general_asr_enabled: bool) -> bool:
    if args.weather_asr is not None:
        return bool(args.weather_asr)
    if "ALFRED_WEATHER_ASR" in os.environ:
        return environment_flag("ALFRED_WEATHER_ASR")
    return general_asr_enabled


def benchmark_enabled(args: argparse.Namespace) -> bool:
    return bool(args.benchmark or environment_flag("ALFRED_BENCHMARK"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Alfred on-device voice assistant")
    parser.add_argument(
        "service_command",
        nargs="?",
        choices=("run", "start", "stop", "restart", "status", "logs"),
        help="Control Alfred's Raspberry Pi user services as one unit",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--set-wayne-manor-url",
        metavar="URL",
        help=("Persist the Wayne Manor API URL in .env; may be combined with run/start/restart"),
    )
    parser.add_argument("--self-check", action="store_true")
    database_group = parser.add_mutually_exclusive_group()
    database_group.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize or validate the local reminder database, then exit",
    )
    database_group.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete all reminders, recreate or validate the schema, then exit",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--configure-audio-match",
        metavar="NAME",
        help="Select matching PortAudio input/output indexes in the configuration, then exit",
    )
    parser.add_argument("--once", action="store_true", help="Exit after one primary command")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the physical audio/ML benchmark path with feedback and actions disabled",
    )
    parser.add_argument("--headless-feedback", action="store_true")
    parser.add_argument(
        "--enable-asr",
        action="store_true",
        help=(
            "Enable optional Moonshine follow-up transcription. Without this flag, "
            "command understanding uses only the VCM."
        ),
    )
    weather_asr_group = parser.add_mutually_exclusive_group()
    weather_asr_group.add_argument(
        "--weather-asr",
        dest="weather_asr",
        action="store_true",
        help="Use Moonshine only for the WEATHER location follow-up (independent of general ASR)",
    )
    weather_asr_group.add_argument(
        "--no-weather-asr",
        dest="weather_asr",
        action="store_false",
        help="Use the configured default weather location without asking for one",
    )
    parser.set_defaults(weather_asr=None)
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock Spotify, calls, Google Chat messages, timers, and alarms for local development",
    )
    parser.add_argument(
        "--call-backend",
        choices=("linphone", "wayne_manor"),
        help="Use Linphone or the Wayne Manor telephone for CALL",
    )
    parser.add_argument("--no-tts", action="store_true", help="Disable spoken TTS responses")
    parser.add_argument(
        "--vcm-mode",
        choices=("mock", "onnx"),
        help="Override the configured VCM for this run without editing files",
    )
    parser.add_argument(
        "--vcm-variant",
        choices=("current", "3a2a"),
        help="Select a packaged ONNX VCM variant; default is current",
    )
    parser.add_argument("--mock-intent")
    parser.add_argument("--mock-decision", choices=("execute", "low_confidence", "unsupported"))
    parser.add_argument("--mock-slot", action="append", type=parse_slot, default=[])
    return parser


def control_services(
    command: str,
    *,
    enable_asr: bool = False,
    weather_asr: bool | None = None,
    call_backend: str | None = None,
    vcm_variant: str | None = None,
    benchmark: bool = False,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    """Control Alfred's application services without touching shared audio services."""
    service_asr = enable_asr and not benchmark
    configure_asr = [
        "systemctl",
        "--user",
        "set-environment" if service_asr else "unset-environment",
        "ALFRED_ENABLE_ASR=1" if service_asr else "ALFRED_ENABLE_ASR",
    ]
    configure_weather_asr = (
        ["systemctl", "--user", "unset-environment", "ALFRED_WEATHER_ASR"]
        if weather_asr is None or benchmark
        else [
            "systemctl",
            "--user",
            "set-environment",
            f"ALFRED_WEATHER_ASR={1 if weather_asr else 0}",
        ]
    )
    configure_call_backend = [
        "systemctl",
        "--user",
        "set-environment" if call_backend is not None else "unset-environment",
        f"ALFRED_CALL_BACKEND={call_backend}"
        if call_backend is not None
        else "ALFRED_CALL_BACKEND",
    ]
    configure_vcm_variant = [
        "systemctl",
        "--user",
        "set-environment" if vcm_variant is not None else "unset-environment",
        f"ALFRED_VCM_VARIANT={vcm_variant}" if vcm_variant is not None else "ALFRED_VCM_VARIANT",
    ]
    configure_benchmark = [
        "systemctl",
        "--user",
        "set-environment" if benchmark else "unset-environment",
        "ALFRED_BENCHMARK=1" if benchmark else "ALFRED_BENCHMARK",
    ]
    if command in {"run", "start"}:
        calls = [
            configure_asr,
            configure_weather_asr,
            configure_call_backend,
            configure_vcm_variant,
            configure_benchmark,
            # Restart also starts an inactive unit and guarantees that a mode
            # change takes effect when Alfred is already running.
            ["systemctl", "--user", "restart", "alfred.service"],
        ]
        mode = "Moonshine ASR enabled" if enable_asr else "VCM-only mode"
        if weather_asr is not None:
            mode += "; weather location Moonshine " + ("enabled" if weather_asr else "disabled")
        if vcm_variant is not None:
            mode += f"; VCM variant {vcm_variant}"
        if benchmark:
            mode = "benchmark mode; actions and feedback disabled"
        success_message = f"Alfred is running ({mode})."
    elif command == "stop":
        calls = [
            ["systemctl", "--user", "stop", "alfred.service"],
            ["systemctl", "--user", "unset-environment", "ALFRED_ENABLE_ASR"],
            ["systemctl", "--user", "unset-environment", "ALFRED_WEATHER_ASR"],
            ["systemctl", "--user", "unset-environment", "ALFRED_CALL_BACKEND"],
            ["systemctl", "--user", "unset-environment", "ALFRED_VCM_VARIANT"],
            ["systemctl", "--user", "unset-environment", "ALFRED_BENCHMARK"],
        ]
        success_message = "Alfred is stopped."
    elif command == "restart":
        calls = [
            configure_asr,
            configure_weather_asr,
            configure_call_backend,
            configure_vcm_variant,
            configure_benchmark,
            ["systemctl", "--user", "stop", *APPLICATION_SERVICES],
            ["systemctl", "--user", "start", "alfred.service"],
        ]
        mode = "Moonshine ASR enabled" if enable_asr else "VCM-only mode"
        if weather_asr is not None:
            mode += "; weather location Moonshine " + ("enabled" if weather_asr else "disabled")
        if vcm_variant is not None:
            mode += f"; VCM variant {vcm_variant}"
        if benchmark:
            mode = "benchmark mode; actions and feedback disabled"
        success_message = f"Alfred was restarted ({mode})."
    elif command == "status":
        calls = [["systemctl", "--user", "status", *APPLICATION_SERVICES, "--no-pager"]]
        success_message = None
    elif command == "logs":
        calls = [["journalctl", "--user", "-u", "alfred.service", "-f"]]
        success_message = None
    else:
        raise ValueError(f"Unsupported service command: {command}")

    for arguments in calls:
        completed = runner(arguments, check=False)
        if completed.returncode != 0:
            return int(completed.returncode)
    if success_message is not None:
        print(success_message)
    return 0


def with_cli_overrides(config: AlfredConfig, args: argparse.Namespace) -> AlfredConfig:
    document = copy.deepcopy(config.document)
    model_manifest = copy.deepcopy(config.model_manifest)
    # ASR is deliberately opt-in. This keeps the default and graded execution
    # path limited to wake word + VAD + the VCM, even if an older/custom
    # configuration file still has clarification.enabled set to true.
    document["clarification"]["enabled"] = bool(
        not benchmark_enabled(args) and (args.enable_asr or environment_flag("ALFRED_ENABLE_ASR"))
    )
    call_backend = args.call_backend or os.environ.get("ALFRED_CALL_BACKEND")
    if call_backend is not None:
        if call_backend not in {"linphone", "wayne_manor"}:
            raise ValueError("ALFRED_CALL_BACKEND must be linphone or wayne_manor")
        document["actions"]["call"]["backend"] = call_backend
    if args.no_tts:
        document["tts"]["enabled"] = False
    if args.vcm_mode is not None:
        document["vcm"]["mode"] = args.vcm_mode
    vcm_variant = args.vcm_variant or os.environ.get("ALFRED_VCM_VARIANT")
    if vcm_variant is not None:
        packaged_vcm = model_manifest.get("artifacts", {}).get("vcm")
        variants = packaged_vcm.get("variants") if isinstance(packaged_vcm, dict) else None
        selected = variants.get(vcm_variant) if isinstance(variants, dict) else None
        if not isinstance(selected, dict):
            raise ValueError(f"Unknown packaged VCM variant: {vcm_variant}")
        packaged_vcm.update(copy.deepcopy(selected))
    if args.mock_intent is not None:
        document["vcm"]["mode"] = "mock"
        document["vcm"]["mock_intent"] = args.mock_intent
    if args.mock_decision is not None:
        document["vcm"]["mode"] = "mock"
        document["vcm"]["mock_decision"] = args.mock_decision
        if args.mock_decision == "unsupported":
            document["vcm"]["mock_intent"] = None
    if args.mock_slot:
        document["vcm"]["mode"] = "mock"
        document["vcm"]["mock_slots"] = {name: {"surface": value} for name, value in args.mock_slot}
    return AlfredConfig(
        root=config.root,
        document=document,
        schema=config.schema,
        action_contract=config.action_contract,
        feedback=config.feedback,
        model_manifest=model_manifest,
    )


def artifact_path(config: AlfredConfig, group: str, name: str) -> Path:
    item = config.model_manifest["artifacts"][group]["files"][name]
    return config.local_path(str(item["path"]))


def run_live(config: AlfredConfig, args: argparse.Namespace) -> None:
    audio_config = config.document["audio"]
    wake_config = config.document["wake_word"]
    capture_config = config.document["command_capture"]
    benchmark_mode = benchmark_enabled(args)
    actions = None
    if benchmark_mode:
        LOGGER.info("benchmark_mode=enabled actions=disabled feedback=disabled")
    else:
        actions = build_action_executor(config, mock_device_actions=args.mock)
        actions.check_api_availability(
            float(config.document["runtime"]["api_probe_timeout_seconds"])
        )
        try:
            level = actions.volume.clamp_to_maximum()
            LOGGER.info("system_volume_ready level_percent=%d", level)
        except SystemActionError as error:
            LOGGER.warning("system_volume_startup_clamp_failed reason=%s", error)
    detector = DualWakeWordDetector(
        hey_alfred_model=artifact_path(config, "wake_word", "hey_alfred"),
        im_batman_model=artifact_path(config, "wake_word", "im_batman"),
        melspectrogram_model=artifact_path(config, "wake_word", "melspectrogram"),
        embedding_model=artifact_path(config, "wake_word", "embedding"),
        hey_alfred_threshold=float(wake_config["hey_alfred_threshold"]),
        im_batman_threshold=float(wake_config["im_batman_threshold"]),
        required_hits=int(wake_config["required_hits"]),
        cooldown_seconds=float(wake_config["cooldown_seconds"]),
    )
    vad = SileroVad(
        artifact_path(config, "vad", "silero"),
        frame_samples=int(audio_config["block_samples"]),
    )
    command_capture = CommandCapture(vad, capture_config)
    clarification_capture = None
    speech_recognizer = None
    slot_clarifier = None
    message_follow_up = None
    general_asr_enabled = bool(config.document["clarification"]["enabled"])
    use_weather_asr = False if benchmark_mode else weather_asr_enabled(args, general_asr_enabled)
    if general_asr_enabled or use_weather_asr:
        clarification_capture_config = dict(capture_config)
        clarification_capture_config["maximum_command_seconds"] = float(
            config.document["clarification"]["maximum_response_seconds"]
        )
        clarification_capture = CommandCapture(vad, clarification_capture_config)
        speech_recognizer = build_speech_recognizer(config)
        if general_asr_enabled:
            slot_clarifier = build_slot_clarifier(config, speech_recognizer)
            message_follow_up = MessageFollowUp(config)
        LOGGER.info(
            "asr_mode=enabled engine=moonshine_v2 general=%s weather_location=%s",
            general_asr_enabled,
            use_weather_asr,
        )
    else:
        LOGGER.info("asr_mode=disabled semantic_model=vcm_only")
    feedback = None
    if not benchmark_mode:
        output_device = playback_output_device(audio_config)
        feedback = (
            NullFeedbackPlayer(config.feedback.get("announcements", {}))
            if args.headless_feedback
            else FeedbackPlayer(
                config,
                output_device,
                build_tts(config, output_device),
            )
        )
    benchmark_logger = None
    try:
        benchmark_logger = BenchmarkEventLogger()
        LOGGER.info("benchmark_event_log=%s", benchmark_logger.path)
    except OSError:
        # Logging must not prevent a normal assistant start. The warning makes
        # a missing benchmark record explicit in the service logs.
        LOGGER.exception("benchmark_event_log_unavailable")
        if benchmark_mode:
            raise
    app = AlfredApplication(
        config=config,
        wake_detector=detector,
        command_capture=command_capture,
        clarification_capture=clarification_capture,
        slot_clarifier=slot_clarifier,
        speech_recognizer=speech_recognizer,
        message_follow_up=message_follow_up,
        weather_asr_enabled=use_weather_asr,
        vcm=build_vcm(config),
        feedback=feedback,
        actions=actions,
        benchmark_logger=benchmark_logger,
        benchmark_mode=benchmark_mode,
    )
    microphone = MicrophoneInput(
        sample_rate=int(audio_config["sample_rate"]),
        block_samples=int(audio_config["block_samples"]),
        device=audio_config.get("input_device"),
        queue_blocks=int(audio_config["queue_blocks"]),
    )
    try:
        with microphone:
            app.run(microphone, once=args.once)
    finally:
        if benchmark_logger is not None:
            benchmark_logger.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.set_wayne_manor_url is not None:
            if args.service_command not in {None, "run", "start", "restart"}:
                raise ValueError(
                    "--set-wayne-manor-url may only be used alone or with run/start/restart"
                )
            parsed = urllib.parse.urlparse(args.set_wayne_manor_url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError("Wayne Manor URL must be an HTTP or HTTPS base URL")
            configured_url = args.set_wayne_manor_url.rstrip("/")
            if not configured_url.endswith("/api/v1"):
                raise ValueError("Wayne Manor URL must end with /api/v1")
            config = load_config(args.config)
            update_environment_file(
                config.root / ".env",
                "ALFRED_WAYNE_MANOR_URL",
                configured_url,
            )
            print(f"Configured Wayne Manor API: {configured_url}")
            if args.service_command is None:
                return 0
        if args.service_command is not None:
            return control_services(
                args.service_command,
                enable_asr=args.enable_asr,
                weather_asr=(
                    args.weather_asr
                    if args.weather_asr is not None
                    else (
                        environment_flag("ALFRED_WEATHER_ASR")
                        if "ALFRED_WEATHER_ASR" in os.environ
                        else None
                    )
                ),
                call_backend=args.call_backend,
                vcm_variant=args.vcm_variant,
                benchmark=args.benchmark,
            )
        config = load_config(args.config)
        load_environment_file(config.root / ".env")
        config = with_cli_overrides(config, args)
        logging.basicConfig(
            level=getattr(logging, str(config.document["runtime"]["log_level"]).upper()),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        if args.list_devices:
            print(MicrophoneInput.list_devices())
            return 0
        if args.configure_audio_match:
            devices = MicrophoneInput.list_devices()
            input_index, output_index = matching_device_indices(devices, args.configure_audio_match)
            config_path = (args.config or DEFAULT_CONFIG).expanduser().resolve()
            document = json.loads(config_path.read_text(encoding="utf-8"))
            document["audio"]["input_device"] = input_index
            document["audio"]["output_device"] = output_index
            temporary = config_path.with_suffix(config_path.suffix + ".tmp")
            temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            temporary.replace(config_path)
            print(
                f"Configured audio input {input_index} and output {output_index} "
                f"for {args.configure_audio_match!r}."
            )
            return 0
        if args.init_db:
            database_path = config.local_path(str(config.document["actions"]["database_path"]))
            version = initialize_database(database_path)
            print(f"Initialized reminder database schema {version}: {database_path}")
            return 0
        if args.reset_db:
            database_path = config.local_path(str(config.document["actions"]["database_path"]))
            version = reset_database(database_path)
            print(f"Reset reminder database schema {version}: {database_path}")
            return 0
        if args.self_check:
            if weather_asr_enabled(args, bool(config.document["clarification"]["enabled"])):
                # The health checker validates the configured Moonshine asset
                # when either the general flow or weather-only flow will use it.
                config.document["clarification"]["enabled"] = True
            report = verify_install(
                config,
                require_vcm=config.document["vcm"]["mode"] == "onnx",
                run_models=True,
            )
            print(json.dumps(report, indent=2))
            return 0 if report["ok"] else 1
        run_live(config, args)
    except KeyboardInterrupt:
        print("\nAlfred stopped.")
        return 0
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

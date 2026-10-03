from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from alfred.__main__ import build_parser, control_services, main, with_cli_overrides
from alfred.config import (
    load_config,
    load_environment_file,
    read_json,
    update_environment_file,
)
from alfred.health import verify_install
from alfred.vcm import MockVcm


def test_canonical_contract_and_local_assets() -> None:
    config = load_config()
    assert len(config.intents) == 19
    assert config.slot_by_intent == {
        "TIMER": "duration",
        "ALARM": "time",
        "TEMPERATURE": "degrees",
        "BRIGHTNESS": "percent",
        "COLOR": "color",
        "CREATE_REMINDER": "task",
    }
    assert set(config.action_by_intent) == set(config.intents)
    assert config.action_contract["mode"] == "mixed"
    assert config.document["actions"]["weather"]["location_name"] == "Quezon City"
    assert config.document["actions"]["google_chat"]["webhook_url_environment"] == (
        "ALFRED_GOOGLE_CHAT_WEBHOOK_URL"
    )
    assert len(config.document["actions"]["google_chat"]["fallback_messages"]) == 5
    assert config.action_by_intent["MESSAGE"]["handler"] == "send_message"
    assert config.action_by_intent["PLAY_MUSIC"]["handler"] == "play_music"
    assert config.action_by_intent["PAUSE"]["handler"] == "pause_music"
    assert config.action_by_intent["STOP"]["handler"] == "stop_music"
    assert config.action_by_intent["NEXT"]["handler"] == "next_music"
    assert "spotify" not in config.document["actions"]
    assert "media" not in config.document["actions"]
    assert config.document["actions"]["wayne_manor"]["base_url"] == ("http://127.0.0.1:8765/api/v1")
    assert config.document["actions"]["wayne_manor"]["base_url_environment"] == (
        "ALFRED_WAYNE_MANOR_URL"
    )
    assert config.document["actions"]["call"]["backend"] == "wayne_manor"
    assert config.document["actions"]["call"]["linphone_destination_environment"] == (
        "ALFRED_LINPHONE_DESTINATION"
    )
    assert config.document["audio"]["feedback_gain"] == 0.15
    assert config.document["audio"]["tts_gain"] == 0.15
    assert config.document["audio"]["use_system_default_output"] is True
    assert config.document["command_capture"]["speech_start_timeout_seconds"] == 4.0
    assert config.document["runtime"]["api_probe_timeout_seconds"] == 1.0
    assert config.document["vcm"]["minimum_intent_confidence"] == 0.866007924079895
    vcm_metadata = read_json(
        config.local_path(config.model_manifest["artifacts"]["vcm"]["metadata_path"])
    )
    assert vcm_metadata["minimum_intent_confidence"] == 0.866007924079895
    assert vcm_metadata["scope_threshold"] == 0.9999032967447421
    assert vcm_metadata["calibration_source"] == {
        "minimum_intent_confidence": "ONNX Runtime validation predictions",
        "scope_threshold": "ONNX Runtime validation predictions",
        "temperature": "validation",
        "selection_split": "validation_only",
        "official_test_or_holdout_used": False,
        "operating_point_objective": "maximize correctly executed supported commands",
        "operating_point_constraints": {
            "action_precision_minimum": 0.95,
            "unsupported_execution_rate_maximum": 0.05,
        },
    }
    assert config.document["actions"]["volume"]["linux_maximum_percent"] == 100
    assert config.document["actions"]["reminders"]["spoken_limit"] is None
    assert config.document["tts"]["voice"] == "en_GB-alan-medium"
    assert config.document["tts"]["speed"] == 1.3
    assert config.model_manifest["artifacts"]["tts"]["voice"] == config.document["tts"]["voice"]
    report = verify_install(config)
    artifact_and_feedback = [
        row for row in report["checks"] if not row["check"].startswith("python_module:")
    ]
    assert artifact_and_feedback
    assert all(row["ok"] for row in artifact_and_feedback)


def test_action_allowlist_enables_all_canonical_intents() -> None:
    config = load_config()
    assert set(config.document["actions"]["enabled_intents"]) == set(config.intents)
    assert config.document["clarification"]["enabled"] is False
    assert config.document["clarification"]["asr"]["engine"] == ("sherpa_onnx_moonshine_v2")
    assert config.action_by_intent["CALL"]["handler"] == "start_call"
    assert config.action_by_intent["TIMER"]["handler"] == "set_timer"
    assert config.action_by_intent["ALARM"]["handler"] == "set_alarm"


def test_cli_can_switch_between_mock_and_real_vcm_without_editing_config() -> None:
    config = load_config()
    parser = build_parser()
    mock = with_cli_overrides(config, parser.parse_args(["--vcm-mode", "mock"]))
    real = with_cli_overrides(config, parser.parse_args(["--vcm-mode", "onnx"]))
    assert mock.document["vcm"]["mode"] == "mock"
    assert real.document["vcm"]["mode"] == "onnx"


def test_cli_can_select_the_packaged_3a2a_vcm_variant(monkeypatch) -> None:
    monkeypatch.delenv("ALFRED_VCM_VARIANT", raising=False)
    config = load_config()
    selected = with_cli_overrides(config, build_parser().parse_args(["--vcm-variant", "3a2a"]))
    artifact = selected.model_manifest["artifacts"]["vcm"]
    assert artifact["bundle_id"] == "3a2a4c8540bd-67de9defe703"
    assert artifact["metadata_path"].startswith("assets/models/vcm/3a2a")


def test_vcm_variant_can_be_selected_through_service_environment(monkeypatch) -> None:
    monkeypatch.setenv("ALFRED_VCM_VARIANT", "3a2a")
    selected = with_cli_overrides(load_config(), build_parser().parse_args([]))
    assert selected.model_manifest["artifacts"]["vcm"]["bundle_id"].startswith("3a2a")


def test_asr_is_disabled_by_default_and_requires_an_explicit_flag(monkeypatch) -> None:
    monkeypatch.delenv("ALFRED_ENABLE_ASR", raising=False)
    config = load_config()
    parser = build_parser()

    vcm_only = with_cli_overrides(config, parser.parse_args([]))
    asr_enabled = with_cli_overrides(config, parser.parse_args(["--enable-asr"]))

    assert vcm_only.document["clarification"]["enabled"] is False
    assert asr_enabled.document["clarification"]["enabled"] is True


def test_cli_default_overrides_legacy_configs_that_enabled_asr(monkeypatch) -> None:
    monkeypatch.delenv("ALFRED_ENABLE_ASR", raising=False)
    config = load_config()
    config.document["clarification"]["enabled"] = True

    resolved = with_cli_overrides(config, build_parser().parse_args([]))

    assert resolved.document["clarification"]["enabled"] is False


def test_systemd_environment_enables_asr_inside_the_service(monkeypatch) -> None:
    monkeypatch.setenv("ALFRED_ENABLE_ASR", "1")

    resolved = with_cli_overrides(load_config(), build_parser().parse_args([]))

    assert resolved.document["clarification"]["enabled"] is True


def test_cli_mock_flag_is_independent_of_the_vcm_mode() -> None:
    args = build_parser().parse_args(["--mock"])
    assert args.mock is True
    assert args.vcm_mode is None


def test_benchmark_flag_parses_for_direct_and_service_runs() -> None:
    parser = build_parser()
    direct = parser.parse_args(["--benchmark"])
    service = parser.parse_args(["run", "--benchmark"])
    assert direct.benchmark is True
    assert direct.service_command is None
    assert service.benchmark is True
    assert service.service_command == "run"


def test_benchmark_environment_disables_optional_asr(monkeypatch) -> None:
    monkeypatch.setenv("ALFRED_BENCHMARK", "1")
    monkeypatch.setenv("ALFRED_ENABLE_ASR", "1")

    resolved = with_cli_overrides(load_config(), build_parser().parse_args([]))

    assert resolved.document["clarification"]["enabled"] is False


def test_call_backend_can_be_selected_by_cli_or_service_environment(monkeypatch) -> None:
    monkeypatch.delenv("ALFRED_CALL_BACKEND", raising=False)
    cli = with_cli_overrides(
        load_config(), build_parser().parse_args(["--call-backend", "linphone"])
    )
    assert cli.document["actions"]["call"]["backend"] == "linphone"

    monkeypatch.setenv("ALFRED_CALL_BACKEND", "wayne_manor")
    service = with_cli_overrides(load_config(), build_parser().parse_args([]))
    assert service.document["actions"]["call"]["backend"] == "wayne_manor"


def test_explicit_mock_slots_are_treated_as_confident_test_fixtures() -> None:
    config = load_config()
    args = build_parser().parse_args(
        ["--mock-intent", "TIMER", "--mock-slot", "duration=ten minutes"]
    )
    mock = MockVcm.from_config(with_cli_overrides(config, args))
    assert mock.result.slot_confidences == {"duration": 1.0}


def test_service_commands_are_part_of_the_python_cli() -> None:
    parser = build_parser()
    assert parser.parse_args(["status"]).service_command == "status"
    assert parser.parse_args(["stop"]).service_command == "stop"


def test_wayne_manor_url_can_be_set_while_starting_the_service() -> None:
    args = build_parser().parse_args(
        ["run", "--set-wayne-manor-url", "http://192.168.1.9:8765/api/v1"]
    )
    assert args.service_command == "run"
    assert args.set_wayne_manor_url == "http://192.168.1.9:8765/api/v1"


def test_wayne_manor_url_command_persists_the_private_setting(
    tmp_path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        "alfred.__main__.load_config",
        lambda path: SimpleNamespace(root=tmp_path),
    )

    assert main(["--set-wayne-manor-url", "http://192.168.1.9:8765/api/v1"]) == 0
    assert (tmp_path / ".env").read_text(encoding="utf-8") == (
        'ALFRED_WAYNE_MANOR_URL="http://192.168.1.9:8765/api/v1"\n'
    )
    assert "Configured Wayne Manor API" in capsys.readouterr().out


def test_wayne_manor_url_command_rejects_an_incomplete_endpoint(capsys) -> None:
    assert main(["--set-wayne-manor-url", "http://192.168.1.9:8765"]) == 2
    assert "must end with /api/v1" in capsys.readouterr().err


def test_service_stop_uses_safe_dependency_order() -> None:
    commands: list[list[str]] = []

    def runner(arguments, **kwargs):
        assert kwargs == {"check": False}
        commands.append(arguments)
        return type("Result", (), {"returncode": 0})()

    assert control_services("stop", runner=runner) == 0
    assert commands == [
        ["systemctl", "--user", "stop", "alfred.service"],
        ["systemctl", "--user", "unset-environment", "ALFRED_ENABLE_ASR"],
        ["systemctl", "--user", "unset-environment", "ALFRED_WEATHER_ASR"],
        ["systemctl", "--user", "unset-environment", "ALFRED_CALL_BACKEND"],
        ["systemctl", "--user", "unset-environment", "ALFRED_VCM_VARIANT"],
        ["systemctl", "--user", "unset-environment", "ALFRED_BENCHMARK"],
    ]


def test_service_run_propagates_asr_mode_and_restarts_alfred() -> None:
    commands: list[list[str]] = []

    def runner(arguments, **kwargs):
        assert kwargs == {"check": False}
        commands.append(arguments)
        return type("Result", (), {"returncode": 0})()

    assert control_services("run", enable_asr=True, runner=runner) == 0
    assert commands == [
        ["systemctl", "--user", "set-environment", "ALFRED_ENABLE_ASR=1"],
        ["systemctl", "--user", "unset-environment", "ALFRED_WEATHER_ASR"],
        ["systemctl", "--user", "unset-environment", "ALFRED_CALL_BACKEND"],
        ["systemctl", "--user", "unset-environment", "ALFRED_VCM_VARIANT"],
        ["systemctl", "--user", "unset-environment", "ALFRED_BENCHMARK"],
        ["systemctl", "--user", "restart", "alfred.service"],
    ]
    assert all("soloist" not in " ".join(command).casefold() for command in commands)


def test_service_run_without_flag_clears_previous_asr_mode() -> None:
    commands: list[list[str]] = []

    def runner(arguments, **kwargs):
        commands.append(arguments)
        return type("Result", (), {"returncode": 0})()

    assert control_services("run", runner=runner) == 0
    assert commands[0] == [
        "systemctl",
        "--user",
        "unset-environment",
        "ALFRED_ENABLE_ASR",
    ]
    assert commands[1] == [
        "systemctl",
        "--user",
        "unset-environment",
        "ALFRED_WEATHER_ASR",
    ]
    assert commands[-1] == ["systemctl", "--user", "restart", "alfred.service"]


def test_service_benchmark_mode_is_preserved_and_suppresses_asr() -> None:
    commands: list[list[str]] = []

    def runner(arguments, **kwargs):
        commands.append(arguments)
        return type("Result", (), {"returncode": 0})()

    assert (
        control_services(
            "run",
            enable_asr=True,
            weather_asr=True,
            benchmark=True,
            runner=runner,
        )
        == 0
    )
    assert ["systemctl", "--user", "unset-environment", "ALFRED_ENABLE_ASR"] in commands
    assert ["systemctl", "--user", "unset-environment", "ALFRED_WEATHER_ASR"] in commands
    assert ["systemctl", "--user", "set-environment", "ALFRED_BENCHMARK=1"] in commands


def test_weather_asr_can_be_enabled_independently_of_general_asr() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--weather-asr"])
    assert args.enable_asr is False
    assert args.weather_asr is True

    commands: list[list[str]] = []

    def runner(arguments, **kwargs):
        commands.append(arguments)
        return type("Result", (), {"returncode": 0})()

    assert control_services("run", weather_asr=True, runner=runner) == 0
    assert commands[0] == [
        "systemctl",
        "--user",
        "unset-environment",
        "ALFRED_ENABLE_ASR",
    ]
    assert commands[1] == [
        "systemctl",
        "--user",
        "set-environment",
        "ALFRED_WEATHER_ASR=1",
    ]


def test_service_run_propagates_call_backend() -> None:
    commands: list[list[str]] = []

    def runner(arguments, **kwargs):
        commands.append(arguments)
        return type("Result", (), {"returncode": 0})()

    assert control_services("run", call_backend="linphone", runner=runner) == 0
    assert commands[2] == [
        "systemctl",
        "--user",
        "set-environment",
        "ALFRED_CALL_BACKEND=linphone",
    ]


def test_service_run_selects_and_clears_the_vcm_variant() -> None:
    commands: list[list[str]] = []

    def runner(arguments, **kwargs):
        commands.append(arguments)
        return type("Result", (), {"returncode": 0})()

    assert control_services("run", vcm_variant="3a2a", runner=runner) == 0
    assert ["systemctl", "--user", "set-environment", "ALFRED_VCM_VARIANT=3a2a"] in commands

    commands.clear()
    assert control_services("run", runner=runner) == 0
    assert ["systemctl", "--user", "unset-environment", "ALFRED_VCM_VARIANT"] in commands


def test_deployment_metadata_example_matches_the_canonical_schema() -> None:
    config = load_config()
    metadata = read_json(config.root / "config" / "vcm_metadata.example.json")
    assert tuple(metadata["intent_labels"]) == config.intents
    assert metadata["slot_by_intent"] == config.slot_by_intent
    assert metadata["slot_tokens"][0] == "<blank>"
    assert metadata["sample_rate"] == config.document["audio"]["sample_rate"]


def test_environment_file_loads_values_without_overriding_process_environment(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# Alfred secrets\n"
        'ALFRED_TEST_QUOTED="https://example.test/path?key=a&token=b"\n'
        "ALFRED_TEST_EXISTING=file-value\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("ALFRED_TEST_QUOTED", raising=False)
    monkeypatch.setenv("ALFRED_TEST_EXISTING", "process-value")
    assert load_environment_file(path)
    assert os.environ["ALFRED_TEST_QUOTED"] == "https://example.test/path?key=a&token=b"
    assert os.environ["ALFRED_TEST_EXISTING"] == "process-value"


def test_environment_file_rejects_shell_syntax(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text("export ALFRED_VALUE=unsafe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid environment name"):
        load_environment_file(path)


def test_environment_file_update_preserves_secrets_and_replaces_one_value(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        'ALFRED_GOOGLE_CHAT_WEBHOOK_URL="https://example.test/secret"\n'
        'ALFRED_WAYNE_MANOR_URL="http://old-host:8765/api/v1"\n',
        encoding="utf-8",
    )

    update_environment_file(
        path,
        "ALFRED_WAYNE_MANOR_URL",
        "http://192.168.1.9:8765/api/v1",
    )

    assert path.read_text(encoding="utf-8") == (
        'ALFRED_GOOGLE_CHAT_WEBHOOK_URL="https://example.test/secret"\n'
        'ALFRED_WAYNE_MANOR_URL="http://192.168.1.9:8765/api/v1"\n'
    )
    assert path.stat().st_mode & 0o777 == 0o600

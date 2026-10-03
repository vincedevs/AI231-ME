from __future__ import annotations

import json

import pytest

from wayne_manor.config import load_config


def test_default_configuration_is_self_consistent() -> None:
    config = load_config()
    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8765
    assert config.default_light_group in config.lights
    assert config.default_thermostat in config.thermostats
    assert config.thermostats[config.default_thermostat].unit == "celsius"
    assert config.telephone.display_name == "Wayne Manor telephone"
    assert config.telephone.ring_duration_seconds == 12
    assert config.telephone.ringtone_url == "/assets/telephone-ring.wav"
    thermostat = config.thermostats[config.default_thermostat]
    assert thermostat.change_sound_url == "/assets/air-conditioner.wav"
    assert thermostat.change_sound_volume == 0.5
    assert config.spotify.client_id == "3fcb96d2b6a84fbdbffff892713e83ca"
    assert config.spotify.redirect_uri == "http://127.0.0.1:8765/"
    assert config.spotify.device_name == "Wayne Manor Turntable"


def test_configuration_rejects_an_unsafe_initial_temperature(tmp_path) -> None:
    source = load_config().path
    document = json.loads(source.read_text(encoding="utf-8"))
    document["thermostats"]["living-room-climate"]["initial_setpoint"] = 50
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="outside its limits"):
        load_config(path)


def test_configuration_rejects_an_unknown_initial_color(tmp_path) -> None:
    source = load_config().path
    document = json.loads(source.read_text(encoding="utf-8"))
    document["lights"]["living-room"]["initial_color"] = "infrared"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="initial_color is not supported"):
        load_config(path)

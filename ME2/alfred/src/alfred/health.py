from __future__ import annotations

import importlib
import importlib.util
from typing import Any

from .config import AlfredConfig, sha256_directory, sha256_file

RUNTIME_MODULES = ("numpy", "onnxruntime", "sounddevice", "soundfile")


def _runtime_action_checks(config: AlfredConfig) -> list[dict[str, Any]]:
    """Check local action prerequisites without changing the system volume."""
    from .reminders import initialize_database
    from .system_actions import VolumeController

    checks: list[dict[str, Any]] = []
    database_path = config.local_path(str(config.document["actions"]["database_path"]))
    try:
        version = initialize_database(database_path)
        checks.append(
            {
                "check": "runtime:reminder_database",
                "ok": True,
                "path": str(database_path),
                "schema_version": version,
            }
        )
    except Exception as error:  # noqa: BLE001 - self-check must report filesystem failures
        checks.append({"check": "runtime:reminder_database", "ok": False, "error": str(error)})

    volume_values = config.document["actions"]["volume"]
    volume = VolumeController(
        step_percent=int(volume_values["step_percent"]),
        linux_maximum_percent=int(volume_values["linux_maximum_percent"]),
        timeout_seconds=float(volume_values["timeout_seconds"]),
    )
    checks.append(
        {
            "check": "runtime:volume_backend",
            "ok": volume.backend is not None,
            "backend": volume.backend,
            "executable": volume.executable,
            "maximum_percent": volume.maximum_percent,
            "note": "Detection only; self-check does not change the volume.",
        }
    )
    return checks


def _runtime_model_checks(config: AlfredConfig, require_vcm: bool) -> list[dict[str, Any]]:
    import numpy as np

    from .vad import SileroVad
    from .vcm import build_vcm
    from .wake_word import DualWakeWordDetector

    checks: list[dict[str, Any]] = []
    artifacts = config.model_manifest["artifacts"]
    try:
        wake_files = artifacts["wake_word"]["files"]
        detector = DualWakeWordDetector(
            hey_alfred_model=config.local_path(wake_files["hey_alfred"]["path"]),
            im_batman_model=config.local_path(wake_files["im_batman"]["path"]),
            melspectrogram_model=config.local_path(wake_files["melspectrogram"]["path"]),
            embedding_model=config.local_path(wake_files["embedding"]["path"]),
            hey_alfred_threshold=0.5,
            im_batman_threshold=0.5,
            required_hits=2,
            cooldown_seconds=0,
        )
        for frame in range(6):
            detector.process(np.zeros(1_280, dtype=np.int16), now=frame * 0.08)
        finite = all(np.isfinite(score) for score in detector.last_scores.values())
        checks.append(
            {"check": "runtime:wake_word", "ok": bool(finite), "scores": detector.last_scores}
        )
    except Exception as error:  # noqa: BLE001 - self-check must report third-party runtime failures
        checks.append({"check": "runtime:wake_word", "ok": False, "error": str(error)})

    try:
        vad_file = artifacts["vad"]["files"]["silero"]["path"]
        vad = SileroVad(config.local_path(vad_file), frame_samples=480)
        score = vad.score(np.zeros(480, dtype=np.float32))
        checks.append(
            {"check": "runtime:vad", "ok": bool(np.isfinite(score)), "silence_score": score}
        )
    except Exception as error:  # noqa: BLE001 - self-check must report third-party runtime failures
        checks.append({"check": "runtime:vad", "ok": False, "error": str(error)})

    if require_vcm:
        try:
            result = build_vcm(config).predict(np.zeros(16_000, dtype=np.float32))
            checks.append(
                {
                    "check": "runtime:vcm",
                    "ok": result.decision.value == "unsupported"
                    and result.intent is None
                    and result.intent_confidence == 0.0,
                    "decision": str(result.decision),
                    "intent": result.intent,
                    "note": "Digital silence must be rejected before ONNX inference.",
                }
            )
        except Exception as error:  # noqa: BLE001 - self-check must report model runtime failures
            checks.append({"check": "runtime:vcm", "ok": False, "error": str(error)})

    if config.document["tts"]["enabled"]:
        try:
            from .tts import build_tts

            tts = build_tts(config)
            if tts is None:
                raise RuntimeError("TTS is enabled but no engine was created")
            samples, sample_rate, metrics = tts.generate("Alfred is ready.")
            checks.append(
                {
                    "check": "runtime:tts",
                    "ok": samples.size > 0 and sample_rate > 0,
                    "sample_rate": sample_rate,
                    **metrics,
                }
            )
        except Exception as error:  # noqa: BLE001 - self-check must report third-party runtime failures
            checks.append({"check": "runtime:tts", "ok": False, "error": str(error)})
    return checks


def verify_install(
    config: AlfredConfig, require_vcm: bool = False, run_models: bool = False
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for module in RUNTIME_MODULES:
        available = importlib.util.find_spec(module) is not None
        checks.append({"check": f"python_module:{module}", "ok": available})
    if config.document["tts"]["enabled"]:
        try:
            importlib.import_module("sherpa_onnx")
        except ImportError as error:
            checks.append({"check": "python_module:sherpa_onnx", "ok": False, "error": str(error)})
        else:
            checks.append({"check": "python_module:sherpa_onnx", "ok": True})

    artifacts = config.model_manifest.get("artifacts", {})
    required_artifacts = ["wake_word", "vad"]
    if config.document["tts"]["enabled"]:
        required_artifacts.append("tts")
    if require_vcm:
        required_artifacts.append("vcm")
    for group_name in required_artifacts:
        group = artifacts.get(group_name)
        if not isinstance(group, dict):
            checks.append({"check": f"artifact_group:{group_name}", "ok": False})
            continue
        files = group.get("files", {})
        if group_name == "vcm" and "path" in group:
            files = {
                "model": {"path": group["path"], "sha256": group.get("sha256")},
                "metadata": {
                    "path": group["metadata_path"],
                    "sha256": group.get("metadata_sha256"),
                },
            }
        for name, item in files.items():
            path = config.local_path(str(item["path"]))
            expected = item.get("sha256")
            actual = sha256_file(path) if path.is_file() else None
            checks.append(
                {
                    "check": f"artifact:{group_name}:{name}",
                    "ok": actual is not None and (expected is None or actual == expected),
                    "path": str(path),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )
        for name, item in group.get("directories", {}).items():
            path = config.local_path(str(item["path"]))
            expected = item.get("sha256")
            expected_count = item.get("file_count")
            if path.is_dir():
                actual, actual_count = sha256_directory(path)
            else:
                actual, actual_count = None, None
            checks.append(
                {
                    "check": f"artifact:{group_name}:{name}",
                    "ok": actual is not None
                    and (expected is None or actual == expected)
                    and (expected_count is None or actual_count == expected_count),
                    "path": str(path),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "expected_file_count": expected_count,
                    "actual_file_count": actual_count,
                }
            )

    if run_models and config.document["clarification"]["enabled"]:
        asr = config.document["clarification"]["asr"]
        for name in ("encoder_path", "decoder_path", "tokens_path"):
            path = config.local_path(str(asr[name]))
            checks.append(
                {
                    "check": f"artifact:clarification_asr:{name}",
                    "ok": path.is_file(),
                    "path": str(path),
                }
            )

    for event, value in config.feedback.get("sounds", {}).items():
        path = config.local_path(str(value))
        checks.append({"check": f"feedback:{event}", "ok": path.is_file(), "path": str(path)})

    prerequisites_ok = all(row["ok"] for row in checks)
    if run_models and prerequisites_ok:
        checks.extend(_runtime_action_checks(config))
        checks.extend(_runtime_model_checks(config, require_vcm))

    return {"ok": all(row["ok"] for row in checks), "checks": checks}

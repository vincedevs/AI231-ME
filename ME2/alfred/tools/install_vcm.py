#!/usr/bin/env python3
"""Validate and transactionally install one completed VCM experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

ALFRED_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ALFRED_ROOT / "src"))

from alfred.config import DEFAULT_CONFIG, AlfredConfig, load_config, sha256_file
from alfred.health import verify_install
from alfred.vcm import DEFAULT_SLOT_TOKENS, OnnxVcm


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Required experiment file is missing: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def build_legacy_metadata(
    experiment: Path,
    config: AlfredConfig,
    minimum_intent_confidence: float | None,
) -> dict[str, Any]:
    """Recover metadata from experiment outputs made before automatic export."""
    status = read_json(experiment / "status.json")
    if status.get("state") != "complete":
        raise ValueError(f"Experiment is not complete: state={status.get('state')!r}")
    experiment_config = read_json(experiment / "config.json")
    metrics = read_json(experiment / "metrics.json")
    identity = experiment_config.get("experiment")
    if not isinstance(identity, dict):
        identity = {
            name: metrics[name]
            for name in (
                "architecture",
                "data_condition",
                "rejection_strategy",
                "protocol",
                "seed",
            )
        }
    evaluation = experiment_config.get("evaluation", {})
    configured_minimum = (
        minimum_intent_confidence
        if minimum_intent_confidence is not None
        else evaluation.get(
            "minimum_intent_confidence",
            config.document["vcm"]["minimum_intent_confidence"],
        )
    )
    test_metrics = metrics["test"]
    return {
        "metadata_version": "1.0.0",
        "schema_version": str(config.schema["schema_version"]),
        "sample_rate": int(experiment_config["audio"]["sample_rate"]),
        "intent_labels": list(config.intents),
        "slot_by_intent": config.slot_by_intent,
        "slot_tokens": list(DEFAULT_SLOT_TOKENS),
        "rejection_strategy": str(identity["rejection_strategy"]),
        # These test-report fields contain values fitted on validation and then
        # held fixed during test evaluation; they are not fitted on test data.
        "temperature": float(test_metrics["temperature"]),
        "scope_threshold": float(test_metrics["scope"]["threshold"]),
        "minimum_intent_confidence": float(configured_minimum),
        "minimum_samples": int(experiment_config["audio"]["n_fft"]),
        "calibration_source": {
            "temperature": "validation_checkpoint",
            "scope_threshold": "validation_checkpoint",
            "minimum_intent_confidence": "configured_before_test_evaluation",
        },
        "experiment": identity,
    }


def resolve_metadata(
    experiment: Path,
    explicit_path: Path | None,
    config: AlfredConfig,
    minimum_intent_confidence: float | None,
) -> tuple[dict[str, Any], str]:
    path = explicit_path or experiment / "deployment_metadata.json"
    if path.is_file():
        metadata = read_json(path)
        source = str(path)
        if minimum_intent_confidence is not None:
            metadata = copy.deepcopy(metadata)
            metadata["minimum_intent_confidence"] = minimum_intent_confidence
            metadata.setdefault("calibration_source", {})["minimum_intent_confidence"] = (
                "installer_override"
            )
        return metadata, source
    if explicit_path is not None:
        raise FileNotFoundError(f"Explicit metadata file does not exist: {explicit_path}")
    return (
        build_legacy_metadata(experiment, config, minimum_intent_confidence),
        "generated from legacy experiment outputs",
    )


def validate_metadata(metadata: dict[str, Any], config: AlfredConfig) -> None:
    if str(metadata.get("schema_version")) != str(config.schema["schema_version"]):
        raise ValueError("VCM metadata schema version does not match Alfred")
    if int(metadata.get("sample_rate", 0)) != int(config.document["audio"]["sample_rate"]):
        raise ValueError("VCM metadata sample rate does not match Alfred")
    if tuple(metadata.get("intent_labels", ())) != config.intents:
        raise ValueError("VCM metadata intent order does not match Alfred")
    if metadata.get("slot_by_intent") != config.slot_by_intent:
        raise ValueError("VCM metadata slot mapping does not match Alfred")
    experiment = metadata.get("experiment")
    if isinstance(experiment, dict):
        identity_strategy = experiment.get("rejection_strategy")
        if identity_strategy != metadata.get("rejection_strategy"):
            raise ValueError("Experiment and deployment rejection strategies disagree")


def validate_model(
    model_path: Path,
    metadata_path: Path,
    config: AlfredConfig,
) -> dict[str, Any]:
    import numpy as np

    model = OnnxVcm(
        model_path,
        metadata_path,
        expected_intents=config.intents,
        expected_slot_by_intent=config.slot_by_intent,
        expected_schema_version=str(config.schema["schema_version"]),
        expected_sample_rate=int(config.document["audio"]["sample_rate"]),
        minimum_waveform_rms=float(config.document["vcm"]["minimum_waveform_rms"]),
        threads=int(config.document["vcm"]["onnx_threads"]),
    )
    result = model.predict(np.zeros(16_000, dtype=np.float32))
    if result.decision.value != "unsupported":
        raise ValueError("VCM deployment safety gate did not reject digital silence")
    return {
        "decision": str(result.decision),
        "intent": result.intent,
        "intent_confidence": result.intent_confidence,
        "in_scope_score": result.in_scope_score,
    }


def atomic_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def install_bundle(
    *,
    source_model: Path,
    metadata: dict[str, Any],
    config: AlfredConfig,
    activate: bool,
) -> dict[str, Any]:
    models_root = config.root / "assets" / "models"
    vcm_root = models_root / "vcm"
    vcm_root.mkdir(parents=True, exist_ok=True)
    stage = vcm_root / f".staging-{uuid.uuid4().hex}"
    stage.mkdir()
    staged_model = stage / "model.onnx"
    staged_metadata = stage / "metadata.json"
    manifest_path = models_root / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    original_config = DEFAULT_CONFIG.read_bytes()
    manifest_committed = False
    config_committed = False
    created_bundle: Path | None = None
    try:
        shutil.copy2(source_model, staged_model)
        staged_metadata.write_bytes(json_bytes(metadata))
        inference = validate_model(staged_model, staged_metadata, config)
        model_hash = sha256_file(staged_model)
        metadata_hash = sha256_file(staged_metadata)
        bundle_id = f"{model_hash[:12]}-{metadata_hash[:12]}"
        destination = vcm_root / bundle_id
        if destination.exists():
            if (
                sha256_file(destination / "model.onnx") != model_hash
                or sha256_file(destination / "metadata.json") != metadata_hash
            ):
                raise RuntimeError(f"Existing VCM bundle is inconsistent: {destination}")
            shutil.rmtree(stage)
        else:
            stage.replace(destination)
            created_bundle = destination

        relative = destination.relative_to(config.root)
        manifest = copy.deepcopy(config.model_manifest)
        manifest.setdefault("artifacts", {})["vcm"] = {
            "bundle_id": bundle_id,
            "path": str(relative / "model.onnx"),
            "sha256": model_hash,
            "size_bytes": (destination / "model.onnx").stat().st_size,
            "metadata_path": str(relative / "metadata.json"),
            "metadata_sha256": metadata_hash,
            "metadata_size_bytes": (destination / "metadata.json").stat().st_size,
        }
        atomic_bytes(manifest_path, json_bytes(manifest))
        manifest_committed = True

        if activate:
            document = copy.deepcopy(config.document)
            document["vcm"]["mode"] = "onnx"
            document["vcm"]["minimum_intent_confidence"] = float(
                metadata["minimum_intent_confidence"]
            )
            atomic_bytes(DEFAULT_CONFIG, json_bytes(document))
            config_committed = True

        installed_config = load_config()
        verification = verify_install(
            installed_config,
            require_vcm=True,
            run_models=activate,
        )
        if not verification["ok"]:
            failed = [row for row in verification["checks"] if not row["ok"]]
            raise RuntimeError(f"Installed bundle failed verification: {failed}")
        return {
            "bundle_id": bundle_id,
            "model": str(destination / "model.onnx"),
            "metadata": str(destination / "metadata.json"),
            "activated": activate,
            "inference_smoke_test": inference,
            "verification_checks": len(verification["checks"]),
        }
    except Exception:
        if config_committed:
            atomic_bytes(DEFAULT_CONFIG, original_config)
        if manifest_committed:
            atomic_bytes(manifest_path, original_manifest)
        if created_bundle is not None:
            shutil.rmtree(created_bundle, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Defaults to EXPERIMENT/model.onnx")
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Defaults to EXPERIMENT/deployment_metadata.json or legacy reconstruction",
    )
    parser.add_argument("--minimum-intent-confidence", type=float)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.activate and args.dry_run:
        parser.error("--activate and --dry-run cannot be used together")
    if args.minimum_intent_confidence is not None and not (
        0 <= args.minimum_intent_confidence <= 1
    ):
        parser.error("--minimum-intent-confidence must be between 0 and 1")
    return args


def main() -> int:
    args = parse_args()
    experiment = args.experiment.expanduser().resolve()
    if not experiment.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment}")
    status = read_json(experiment / "status.json")
    if status.get("state") != "complete":
        raise ValueError(f"Experiment is not complete: state={status.get('state')!r}")
    model = (args.model or experiment / "model.onnx").expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(f"VCM ONNX model does not exist: {model}")
    config = load_config()
    metadata_path = args.metadata.expanduser().resolve() if args.metadata else None
    metadata, metadata_source = resolve_metadata(
        experiment,
        metadata_path,
        config,
        args.minimum_intent_confidence,
    )
    validate_metadata(metadata, config)

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="alfred-vcm-check-") as directory:
            temporary_metadata = Path(directory) / "metadata.json"
            temporary_metadata.write_bytes(json_bytes(metadata))
            inference = validate_model(model, temporary_metadata, config)
        report = {
            "valid": True,
            "dry_run": True,
            "model": str(model),
            "metadata_source": metadata_source,
            "experiment": metadata.get("experiment"),
            "inference_smoke_test": inference,
        }
    else:
        report = install_bundle(
            source_model=model,
            metadata=metadata,
            config=config,
            activate=args.activate,
        )
        report["metadata_source"] = metadata_source
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

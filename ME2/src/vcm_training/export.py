from __future__ import annotations

import copy
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch


def build_deployment_metadata(
    *,
    audio_config: dict[str, Any],
    intent_labels: tuple[str, ...],
    slot_by_intent: dict[str, str],
    slot_tokens: tuple[str, ...],
    rejection_strategy: str,
    temperature: float,
    scope_threshold: float,
    minimum_intent_confidence: float,
    experiment_identity: dict[str, Any],
) -> dict[str, Any]:
    """Build the application-side contract using validation-derived calibration."""
    if rejection_strategy not in {"confidence", "unknown_class", "binary_scope"}:
        raise ValueError(f"Unsupported rejection strategy: {rejection_strategy}")
    if temperature <= 0:
        raise ValueError("Deployment temperature must be positive")
    if not 0 <= scope_threshold <= 1:
        raise ValueError("Deployment scope threshold must be between 0 and 1")
    if not 0 <= minimum_intent_confidence <= 1:
        raise ValueError("Minimum intent confidence must be between 0 and 1")
    return {
        "metadata_version": "1.0.0",
        "schema_version": "1.0.0",
        "sample_rate": int(audio_config["sample_rate"]),
        "intent_labels": list(intent_labels),
        "slot_by_intent": dict(slot_by_intent),
        "slot_tokens": list(slot_tokens),
        "rejection_strategy": rejection_strategy,
        "temperature": float(temperature),
        "scope_threshold": float(scope_threshold),
        "minimum_intent_confidence": float(minimum_intent_confidence),
        "minimum_samples": int(audio_config["n_fft"]),
        "calibration_source": {
            "temperature": "validation",
            "scope_threshold": "validation",
            "minimum_intent_confidence": "configured_before_test_evaluation",
        },
        "experiment": dict(experiment_identity),
    }


def export_onnx(
    model: torch.nn.Module,
    destination: Path,
    audio_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    """Export the raw-waveform model and verify ONNX Runtime numerical parity."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported = copy.deepcopy(model).cpu().eval()
    generator = torch.Generator().manual_seed(231)
    trace_samples = int(audio_config["sample_rate"] * 2)
    waveform = torch.randn(1, trace_samples, generator=generator) * 0.05
    lengths = torch.tensor([trace_samples], dtype=torch.long)
    torch.onnx.export(
        exported,
        (waveform, lengths),
        destination,
        input_names=["waveform", "lengths"],
        output_names=["intent_logits", "slot_logits", "scope_logits", "sequence_lengths"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "samples"},
            "lengths": {0: "batch"},
            "intent_logits": {0: "batch"},
            "slot_logits": {0: "batch", 1: "frames"},
            "scope_logits": {0: "batch"},
            "sequence_lengths": {0: "batch"},
        },
        opset_version=int(evaluation_config["onnx_opset"]),
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(destination)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    comparisons = []
    validation_seconds = tuple(evaluation_config.get("onnx_validation_seconds", (1.0, 2.0, 3.0)))
    for seconds in validation_seconds:
        samples = int(audio_config["sample_rate"] * float(seconds))
        check_waveform = torch.randn(1, samples, generator=generator) * 0.05
        check_lengths = torch.tensor([samples], dtype=torch.long)
        with torch.inference_mode():
            expected_outputs = exported(check_waveform, check_lengths)
        actual_outputs = session.run(
            None,
            {
                "waveform": check_waveform.numpy(),
                "lengths": check_lengths.numpy(),
            },
        )
        for name, expected, observed in zip(
            ("intent_logits", "slot_logits", "scope_logits", "sequence_lengths"),
            expected_outputs,
            actual_outputs,
            strict=True,
        ):
            expected_array = expected.detach().numpy()
            absolute_error = float(np.max(np.abs(expected_array - observed)))
            close = bool(
                np.allclose(
                    expected_array,
                    observed,
                    atol=float(evaluation_config["onnx_absolute_tolerance"]),
                    rtol=float(evaluation_config["onnx_relative_tolerance"]),
                )
            )
            comparisons.append(
                {
                    "audio_seconds": float(seconds),
                    "output": name,
                    "maximum_absolute_error": absolute_error,
                    "close": close,
                }
            )
    if not all(item["close"] for item in comparisons):
        raise RuntimeError(f"ONNX parity check failed: {comparisons}")

    inputs = {"waveform": waveform.numpy(), "lengths": lengths.numpy()}
    for _ in range(int(evaluation_config["latency_warmup_runs"])):
        session.run(None, inputs)
    latencies = []
    for _ in range(int(evaluation_config["latency_timed_runs"])):
        started = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "path": str(destination),
        "size_bytes": destination.stat().st_size,
        "parameter_count": sum(parameter.numel() for parameter in exported.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in exported.parameters() if parameter.requires_grad
        ),
        "input_contract": {
            "waveform": "float32[batch,samples]",
            "lengths": "int64[batch]",
            "sample_rate": int(audio_config["sample_rate"]),
            "minimum_samples": int(audio_config["n_fft"]),
        },
        "parity": comparisons,
        "cpu_latency_two_second_audio_ms": {
            "median": statistics.median(latencies),
            "p95": ordered[p95_index],
            "runs": len(latencies),
            "host_specific": True,
        },
    }

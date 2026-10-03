#!/usr/bin/env python3
"""Benchmark Alfred's local inference stages on the current device.

Run the same command and, ideally, the same 16 kHz mono WAV fixture on every
device being compared. The benchmark does not use the microphone, speaker, or
network APIs, so the results isolate local inference performance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alfred.config import load_config, sha256_file
from alfred.speech_recognition import build_speech_recognizer
from alfred.tts import build_tts
from alfred.vad import SileroVad
from alfred.vcm import build_vcm
from alfred.wake_word import DualWakeWordDetector

BENCHMARK_VERSION = "1.0"
SAMPLE_RATE = 16_000


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated sample percentile."""
    if not values:
        raise ValueError("Cannot calculate a percentile from no measurements")
    if not 0 <= fraction <= 1:
        raise ValueError("Percentile fraction must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(latencies_ms: list[float], audio_seconds: float | None = None) -> dict[str, float]:
    summary = {
        "runs": len(latencies_ms),
        "mean_ms": statistics.fmean(latencies_ms),
        "stdev_ms": statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        "minimum_ms": min(latencies_ms),
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "maximum_ms": max(latencies_ms),
        "operations_per_second": 1000.0 / statistics.fmean(latencies_ms),
    }
    if audio_seconds is not None:
        summary["audio_seconds"] = audio_seconds
        summary["real_time_factor_p50"] = summary["p50_ms"] / (audio_seconds * 1000)
        summary["real_time_factor_p95"] = summary["p95_ms"] / (audio_seconds * 1000)
    return summary


def measure(
    operation: Callable[[], Any],
    *,
    warmup: int,
    runs: int,
    audio_seconds: float | None = None,
) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    latencies_ms: list[float] = []
    for _ in range(runs):
        started = time.perf_counter_ns()
        operation()
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "summary": summarize(latencies_ms, audio_seconds),
        "latencies_ms": latencies_ms,
    }


def timed(operation: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    result = operation()
    return result, (time.perf_counter_ns() - started) / 1_000_000


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def command_output(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def cpu_name() -> str:
    if sys.platform == "darwin":
        return command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    cpuinfo = read_text(Path("/proc/cpuinfo")) or ""
    for key in ("Model", "model name", "Hardware"):
        for line in cpuinfo.splitlines():
            if line.startswith((f"{key}\t", f"{key} ")):
                return line.split(":", 1)[-1].strip()
    return platform.processor() or "unknown"


def thermal_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    temperature = read_text(Path("/sys/class/thermal/thermal_zone0/temp"))
    if temperature:
        try:
            snapshot["cpu_temperature_celsius"] = float(temperature) / 1000
        except ValueError:
            snapshot["cpu_temperature_raw"] = temperature
    throttled = command_output(["vcgencmd", "get_throttled"])
    if throttled is not None:
        snapshot["raspberry_pi_throttled"] = throttled
    return snapshot


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "onnxruntime", "sherpa-onnx", "soundfile"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def generated_waveform(seconds: float) -> Any:
    import numpy as np

    sample_count = round(seconds * SAMPLE_RATE)
    indices = np.arange(sample_count, dtype=np.uint64)
    integer_noise = ((indices * 1_103_515_245 + 12_345) % 65_536).astype(np.int32)
    integer_noise -= 32_768
    # Integer generation is bit-identical across ARM and x86. The result is a
    # non-silent compute workload, not an accuracy fixture.
    return (integer_noise.astype(np.float32) * np.float32(0.03 / 32_768)).astype(np.float32)


def load_waveform(path: Path | None, seconds: float) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if path is None:
        waveform = generated_waveform(seconds)
        return waveform, {
            "kind": "deterministic_generated_waveform",
            "generator_version": 1,
            "sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
            "sample_rate": SAMPLE_RATE,
            "samples": int(waveform.size),
            "duration_seconds": waveform.size / SAMPLE_RATE,
            "accuracy_interpretation": "performance_only",
        }
    import soundfile as sf

    resolved = path.expanduser().resolve()
    waveform, sample_rate = sf.read(resolved, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"Audio fixture must be 16 kHz; received {sample_rate} Hz")
    if waveform.ndim != 1:
        raise ValueError("Audio fixture must be mono")
    if waveform.size == 0 or not bool(np.isfinite(waveform).all()):
        raise ValueError("Audio fixture must contain finite samples")
    waveform = np.asarray(waveform, dtype=np.float32).clip(-1, 1)
    return waveform, {
        "kind": "wav_fixture",
        "filename": resolved.name,
        "file_sha256": sha256_file(resolved),
        "waveform_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
        "sample_rate": SAMPLE_RATE,
        "samples": int(waveform.size),
        "duration_seconds": waveform.size / SAMPLE_RATE,
        "accuracy_interpretation": "identical_fixture_required_for_comparison",
    }


def artifact_path(config: Any, group: str, name: str) -> Path:
    item = config.model_manifest["artifacts"][group]["files"][name]
    return config.local_path(str(item["path"]))


def build_wake_word(config: Any) -> DualWakeWordDetector:
    values = config.document["wake_word"]
    return DualWakeWordDetector(
        hey_alfred_model=artifact_path(config, "wake_word", "hey_alfred"),
        im_batman_model=artifact_path(config, "wake_word", "im_batman"),
        melspectrogram_model=artifact_path(config, "wake_word", "melspectrogram"),
        embedding_model=artifact_path(config, "wake_word", "embedding"),
        hey_alfred_threshold=float(values["hey_alfred_threshold"]),
        im_batman_threshold=float(values["im_batman_threshold"]),
        required_hits=int(values["required_hits"]),
        cooldown_seconds=0,
    )


def model_identity(config: Any) -> dict[str, Any]:
    artifacts = config.model_manifest["artifacts"]
    vcm = artifacts["vcm"]
    return {
        "manifest_sha256": sha256_file(config.root / "assets/models/manifest.json"),
        "vcm_model_sha256": vcm.get("sha256"),
        "vcm_metadata_sha256": vcm.get("metadata_sha256"),
        "vcm_threads": int(config.document["vcm"]["onnx_threads"]),
        "asr_threads": int(config.document["clarification"]["asr"]["threads"]),
        "tts_threads": int(config.document["tts"]["threads"]),
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("label", "stage", "metric", "value", "unit"),
        )
        writer.writeheader()
        for stage_name, stage in report["stages"].items():
            for metric, value in stage.get("steady_state", {}).get("summary", {}).items():
                if not isinstance(value, (int, float)):
                    continue
                unit = "milliseconds" if metric.endswith("_ms") else "ratio"
                if metric == "runs":
                    unit = "count"
                elif metric == "operations_per_second":
                    unit = "operations_per_second"
                elif metric == "audio_seconds":
                    unit = "seconds"
                writer.writerow(
                    {
                        "label": report["label"],
                        "stage": stage_name,
                        "metric": metric,
                        "value": value,
                        "unit": unit,
                    }
                )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    config = load_config(args.config)
    if config.document["vcm"]["mode"] != "onnx":
        raise RuntimeError("Set vcm.mode to 'onnx' before benchmarking the real model")
    waveform, workload = load_waveform(args.audio, args.seconds)
    audio_seconds = waveform.size / SAMPLE_RATE
    wake_samples = np.resize((waveform * 32767).astype(np.int16), 1_280)
    vad_samples = np.resize(waveform, int(config.document["audio"]["block_samples"]))

    report: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "label": args.label,
        "parameters": {"warmup": args.warmup, "runs": args.runs},
        "workload": workload,
        "models": model_identity(config),
        "system": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": cpu_name(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "packages": package_versions(),
            "peak_rss_mib_before": peak_rss_mib(),
            "thermal_before": thermal_snapshot(),
        },
        "stages": {},
    }
    suite_started = time.perf_counter()

    print("[1/5] Wake-word inference", flush=True)
    wake_word, initialization_ms = timed(lambda: build_wake_word(config))
    steady = measure(
        lambda: wake_word.process(wake_samples),
        warmup=args.warmup,
        runs=args.runs,
        audio_seconds=1_280 / SAMPLE_RATE,
    )
    report["stages"]["wake_word"] = {
        "initialization_ms": initialization_ms,
        "steady_state": steady,
        "peak_rss_mib_after": peak_rss_mib(),
        "last_scores": wake_word.last_scores,
    }

    print("[2/5] Silero VAD inference", flush=True)
    vad, initialization_ms = timed(
        lambda: SileroVad(
            artifact_path(config, "vad", "silero"),
            frame_samples=int(config.document["audio"]["block_samples"]),
        )
    )
    steady = measure(
        lambda: vad.score(vad_samples),
        warmup=args.warmup,
        runs=args.runs,
        audio_seconds=vad.frame_samples / SAMPLE_RATE,
    )
    report["stages"]["silero_vad"] = {
        "initialization_ms": initialization_ms,
        "steady_state": steady,
        "peak_rss_mib_after": peak_rss_mib(),
    }

    print("[3/5] VCM inference", flush=True)
    vcm, initialization_ms = timed(lambda: build_vcm(config))
    steady = measure(
        lambda: vcm.predict(waveform),
        warmup=args.warmup,
        runs=args.runs,
        audio_seconds=audio_seconds,
    )
    vcm_result = vcm.predict(waveform)
    report["stages"]["vcm"] = {
        "initialization_ms": initialization_ms,
        "steady_state": steady,
        "peak_rss_mib_after": peak_rss_mib(),
        "output": {
            "decision": vcm_result.decision.value,
            "intent": vcm_result.intent,
            "intent_confidence": vcm_result.intent_confidence,
            "in_scope_score": vcm_result.in_scope_score,
            "slots": vcm_result.slots,
        },
    }

    print("[4/5] Moonshine ASR inference", flush=True)
    if args.skip_asr or not bool(config.document["clarification"]["enabled"]):
        report["stages"]["moonshine_asr"] = {"status": "skipped"}
    else:
        recognizer = build_speech_recognizer(config)
        if recognizer is None:
            raise RuntimeError("Clarification ASR is disabled")
        transcript, cold_start_ms = timed(lambda: recognizer.transcribe(waveform))
        steady = measure(
            lambda: recognizer.transcribe(waveform),
            warmup=args.warmup,
            runs=args.runs,
            audio_seconds=audio_seconds,
        )
        report["stages"]["moonshine_asr"] = {
            "cold_start_ms": cold_start_ms,
            "steady_state": steady,
            "peak_rss_mib_after": peak_rss_mib(),
            "transcript": transcript,
        }

    print("[5/5] Piper TTS inference", flush=True)
    if args.skip_tts:
        report["stages"]["piper_tts"] = {"status": "skipped"}
    else:
        tts, initialization_ms = timed(lambda: build_tts(config))
        if tts is None:
            raise RuntimeError("TTS is disabled")
        tts_text = "Alfred is ready to assist you."
        generated, cold_start_ms = timed(lambda: tts.generate(tts_text))
        steady = measure(
            lambda: tts.generate(tts_text),
            warmup=args.warmup,
            runs=args.runs,
            audio_seconds=float(generated[2]["audio_seconds"]),
        )
        report["stages"]["piper_tts"] = {
            "initialization_ms": initialization_ms,
            "cold_start_inference_ms": cold_start_ms,
            "steady_state": steady,
            "peak_rss_mib_after": peak_rss_mib(),
            "text": tts_text,
            "generated_audio_seconds": generated[2]["audio_seconds"],
            "sample_rate": generated[1],
        }

    report["total_elapsed_seconds"] = time.perf_counter() - suite_started
    report["system"]["peak_rss_mib_after"] = peak_rss_mib()
    report["system"]["thermal_after"] = thermal_snapshot()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--label", default=platform.node() or platform.system())
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional 16 kHz mono WAV used for VCM and ASR; use the same file on both devices",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="Generated workload duration when --audio is omitted",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "benchmark_results")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.seconds <= 0 or args.warmup < 0 or args.runs <= 0:
        parser.error("seconds and runs must be positive; warmup cannot be negative")
    report = run_benchmark(args)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_label = "".join(character if character.isalnum() else "_" for character in args.label)
    output = args.output_dir.expanduser().resolve() / f"{safe_label}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    json_path = output / "benchmark.json"
    csv_path = output / "measurements.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(report, csv_path)
    print(f"Benchmark JSON: {json_path}")
    print(f"Measurements CSV: {csv_path}")
    print(f"Peak RSS: {report['system']['peak_rss_mib_after']:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

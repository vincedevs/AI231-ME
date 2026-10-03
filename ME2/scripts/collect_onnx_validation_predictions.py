"""Collect deployment-equivalent VCM predictions from a validation manifest only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_waveform(path: Path, sample_rate: int, maximum_seconds: float, minimum_samples: int) -> np.ndarray:
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    values = np.mean(waveform, axis=1, dtype=np.float32)
    if int(source_rate) != sample_rate:
        divisor = math.gcd(int(source_rate), sample_rate)
        values = resample_poly(values, sample_rate // divisor, int(source_rate) // divisor)
    values = values[: round(sample_rate * maximum_seconds)]
    if not values.size:
        raise ValueError(f"Empty audio file: {path}")
    if values.size < minimum_samples:
        values = np.pad(values, (0, minimum_samples - values.size))
    return np.asarray(values, dtype=np.float32)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--prepared-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "validation" not in args.output.name:
        raise ValueError("Output name must identify this as validation-only predictions")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    records = read_jsonl(args.prepared_dataset / "manifests" / "validation.jsonl")
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, args.threads)
    session = ort.InferenceSession(
        str(args.model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    output_names = [item.name for item in session.get_outputs()]
    intent_labels = list(metadata["intent_labels"])
    temperature = float(metadata["temperature"])
    sample_rate = int(metadata["sample_rate"])
    minimum_samples = int(metadata["minimum_samples"])
    maximum_seconds = 12.0
    predictions = []
    for row in records:
        waveform = load_waveform(
            args.prepared_dataset / row["audio_path"],
            sample_rate,
            maximum_seconds,
            minimum_samples,
        )
        outputs = dict(
            zip(
                output_names,
                session.run(
                    None,
                    {
                        "waveform": waveform[None, :],
                        "lengths": np.asarray([waveform.size], dtype=np.int64),
                    },
                ),
                strict=True,
            )
        )
        probabilities = softmax(np.asarray(outputs["intent_logits"])[0] / temperature)
        intent_index = int(np.argmax(probabilities))
        scope_logit = float(np.asarray(outputs["scope_logits"])[0].reshape(-1)[0])
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "supported": bool(row["supported"]),
                "target_intent": row["canonical_intent"],
                "predicted_intent": intent_labels[intent_index],
                "in_scope_score": float(1.0 / (1.0 + np.exp(-scope_logit))),
                "intent_confidence": float(probabilities[intent_index]),
                "prediction_engine": "onnxruntime_cpu",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(predictions),
                "split": "validation_only",
                "engine": "onnxruntime_cpu",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

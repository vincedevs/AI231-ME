#!/usr/bin/env python3
"""Evaluate the selected ONNX VCM once on the sealed Raspberry Pi holdout."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, help="Override a non-portable model path in selection.json"
    )
    parser.add_argument("--metadata", type=Path, help="Override the deployment metadata path")
    parser.add_argument("--prepared-dataset", type=Path, default=PROJECT_ROOT / "dataset/prepared")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--confirm-final",
        action="store_true",
        help="Required acknowledgement that this consumes the sealed final holdout",
    )
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def main() -> int:
    args = parse_args()
    if not args.confirm_final:
        raise SystemExit("Refusing to open the sealed holdout without --confirm-final")
    if args.batch_size < 1 or args.threads < 1:
        raise ValueError("--batch-size and --threads must be positive")

    import onnxruntime as ort
    import torch
    from torch.utils.data import DataLoader

    from dataset_pipeline.common import sha256_file, write_json
    from dataset_pipeline.prepare import read_jsonl
    from vcm_training.data import CommandDataset, collate_commands
    from vcm_training.metrics import greedy_ctc_decode
    from vcm_training.training import evaluate_collected

    selection_path = args.selection.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model_path = args.model or Path(selection["model"])
    metadata_path = args.metadata or Path(selection["metadata"])
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    if not metadata_path.is_absolute():
        metadata_path = (PROJECT_ROOT / metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    run_directory = model_path.parent
    run_config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))

    prepared = args.prepared_dataset.resolve()
    release_path = prepared / "release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    holdout_path = prepared / "manifests/holdout.jsonl"
    expected_hash = release["manifest_sha256"]["manifests/holdout.jsonl"]
    if sha256_file(holdout_path) != expected_hash:
        raise ValueError("Holdout manifest checksum does not match the prepared release")
    records = read_jsonl(holdout_path)
    dataset = CommandDataset(records, prepared, run_config["audio"], None, int(run_config["seed"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_commands,
    )

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    supported_logits = []
    supported_targets = []
    supported_predicted_slots: list[str] = []
    supported_target_slots: list[str] = []
    all_intent_logits = []
    all_scope_logits = []
    all_supported = []
    sample_ids: list[str] = []
    sources: list[str] = []
    ood_types: list[str] = []
    category_partitions: list[str] = []
    synthetic_flags: list[bool] = []
    latencies_ms: list[float] = []
    audio_seconds = 0.0

    for batch in loader:
        inputs = {
            "waveform": batch["waveform"].numpy().astype(np.float32, copy=False),
            "lengths": batch["lengths"].numpy().astype(np.int64, copy=False),
        }
        started = time.perf_counter()
        intent, slots, scope, sequence_lengths = session.run(None, inputs)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        audio_seconds += float(inputs["lengths"].sum()) / int(metadata["sample_rate"])
        intent_tensor = torch.from_numpy(intent)
        slot_tensor = torch.from_numpy(slots)
        scope_tensor = torch.from_numpy(scope)
        length_tensor = torch.from_numpy(sequence_lengths)
        supported = batch["scope_targets"].bool()
        decoded = greedy_ctc_decode(slot_tensor, length_tensor)
        if supported.any():
            supported_logits.append(intent_tensor[supported, : len(metadata["intent_labels"])])
            supported_targets.append(batch["intent_targets"][supported])
            mask = supported.tolist()
            supported_predicted_slots.extend(value for value, keep in zip(decoded, mask) if keep)
            supported_target_slots.extend(
                value for value, keep in zip(batch["slot_texts"], mask) if keep
            )
        all_intent_logits.append(intent_tensor)
        all_scope_logits.append(scope_tensor)
        all_supported.append(supported)
        sample_ids.extend(batch["sample_ids"])
        sources.extend(batch["source_datasets"])
        ood_types.extend(batch["ood_types"])
        category_partitions.extend(batch["category_partitions"])
        synthetic_flags.extend(batch["is_synthetic"])

    collected = {
        "supported_logits": torch.cat(supported_logits),
        "supported_targets": torch.cat(supported_targets),
        "supported_predicted_slots": supported_predicted_slots,
        "supported_target_slots": supported_target_slots,
        "all_intent_logits": torch.cat(all_intent_logits),
        "all_scope_logits": torch.cat(all_scope_logits),
        "all_supported": torch.cat(all_supported),
        "sample_ids": sample_ids,
        "sources": sources,
        "ood_types": ood_types,
        "category_partitions": category_partitions,
    }
    def select_collected(mask: torch.Tensor) -> dict:
        supported_mask = mask[collected["all_supported"]]
        indices = mask.nonzero(as_tuple=False).flatten().tolist()
        supported_indices = supported_mask.nonzero(as_tuple=False).flatten().tolist()
        return {
            "supported_logits": collected["supported_logits"][supported_mask],
            "supported_targets": collected["supported_targets"][supported_mask],
            "supported_predicted_slots": [
                collected["supported_predicted_slots"][index] for index in supported_indices
            ],
            "supported_target_slots": [
                collected["supported_target_slots"][index] for index in supported_indices
            ],
            "all_intent_logits": collected["all_intent_logits"][mask],
            "all_scope_logits": collected["all_scope_logits"][mask],
            "all_supported": collected["all_supported"][mask],
            "sample_ids": [collected["sample_ids"][index] for index in indices],
            "sources": [collected["sources"][index] for index in indices],
            "ood_types": [collected["ood_types"][index] for index in indices],
            "category_partitions": [
                collected["category_partitions"][index] for index in indices
            ],
        }

    synthetic_mask = torch.tensor(synthetic_flags, dtype=torch.bool)
    strata = {
        "all": torch.ones(len(records), dtype=torch.bool),
        "real": ~synthetic_mask,
        "synthetic": synthetic_mask,
    }
    metrics = {}
    predictions = []
    for name, mask in strata.items():
        stratum_metrics, stratum_predictions, _ = evaluate_collected(
            select_collected(mask),
            str(metadata["rejection_strategy"]),
            run_config["evaluation"],
            float(metadata["temperature"]),
            float(metadata["scope_threshold"]),
        )
        metrics[name] = stratum_metrics
        if name == "all":
            predictions = stratum_predictions
    runtime_seconds = sum(latencies_ms) / 1000.0
    report = {
        "policy": "single final sealed-holdout evaluation",
        "selection_sha256": sha256_file(selection_path),
        "dataset_release_sha256": sha256_file(release_path),
        "holdout_manifest_sha256": expected_hash,
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "records": len(records),
        "metrics": metrics,
        "raspberry_pi_cpu_inference": {
            "batch_size": args.batch_size,
            "threads": args.threads,
            "median_batch_latency_ms": statistics.median(latencies_ms),
            "p95_batch_latency_ms": percentile(latencies_ms, 0.95),
            "total_inference_seconds": runtime_seconds,
            "audio_seconds": audio_seconds,
            "real_time_factor": runtime_seconds / max(audio_seconds, 1e-12),
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "holdout_report.json", report)
    with (args.output / "holdout_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

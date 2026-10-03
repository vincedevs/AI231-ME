from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vcm_training.data import load_experiment_records
from vcm_training.metrics import select_scope_threshold


def write_manifest(root: Path, split: str, rows: list[dict]) -> None:
    path = root / "manifests" / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_rows() -> list[dict]:
    return [
        {
            "sample_id": "benchmark-real",
            "supported": True,
            "canonical_intent": "WEATHER",
            "is_synthetic": False,
            "is_supplemental": False,
        },
        {
            "sample_id": "benchmark-synthetic",
            "supported": True,
            "canonical_intent": "WEATHER",
            "is_synthetic": True,
            "is_supplemental": False,
        },
        {
            "sample_id": "supplement-real",
            "supported": True,
            "canonical_intent": "WEATHER",
            "is_synthetic": False,
            "is_supplemental": True,
        },
        {
            "sample_id": "unsupported",
            "supported": False,
            "canonical_intent": None,
            "is_synthetic": False,
            "is_supplemental": False,
        },
        {
            "sample_id": "supplement-synthetic",
            "supported": True,
            "canonical_intent": "WEATHER",
            "is_synthetic": True,
            "is_supplemental": True,
            "is_supplemental_synthetic": True,
        },
        {
            "sample_id": "synthetic-negative",
            "supported": False,
            "canonical_intent": None,
            "is_synthetic": True,
            "is_supplemental": True,
            "is_negative_ablation": True,
        },
    ]


def test_training_conditions_are_explicit(tmp_path: Path) -> None:
    write_manifest(tmp_path, "train", fixture_rows())
    assert {row["sample_id"] for row in load_experiment_records(
        tmp_path, "train", "benchmark_real", "binary_scope"
    )} == {"benchmark-real", "unsupported"}
    assert {row["sample_id"] for row in load_experiment_records(
        tmp_path, "train", "benchmark_mixed", "binary_scope"
    )} == {"benchmark-real", "benchmark-synthetic", "unsupported"}
    assert {row["sample_id"] for row in load_experiment_records(
        tmp_path, "train", "supplemented_real", "binary_scope"
    )} == {"benchmark-real", "supplement-real", "unsupported"}
    assert {row["sample_id"] for row in load_experiment_records(
        tmp_path, "train", "supplemented_mixed", "binary_scope"
    )} == {
        "benchmark-real",
        "benchmark-synthetic",
        "supplement-real",
        "unsupported",
    }
    assert len(load_experiment_records(
        tmp_path, "train", "supplemented_mixed_extended", "binary_scope"
    )) == 5
    assert {row["sample_id"] for row in load_experiment_records(
        tmp_path, "train", "supplemented_mixed_with_negatives", "binary_scope"
    )} == {
        "benchmark-real",
        "benchmark-synthetic",
        "supplement-real",
        "unsupported",
        "synthetic-negative",
    }


def test_confidence_training_excludes_unsupported_but_evaluation_keeps_it(tmp_path: Path) -> None:
    rows = fixture_rows()
    write_manifest(tmp_path, "train", rows)
    write_manifest(tmp_path, "test", rows)
    train = load_experiment_records(tmp_path, "train", "supplemented_mixed", "confidence")
    test = load_experiment_records(tmp_path, "test", "supplemented_mixed", "confidence")
    assert all(row["supported"] for row in train)
    assert any(not row["supported"] for row in test)


def test_training_loader_cannot_open_sealed_holdout(tmp_path: Path) -> None:
    write_manifest(tmp_path, "holdout", fixture_rows())
    try:
        load_experiment_records(tmp_path, "holdout", "benchmark_real", "binary_scope")
    except ValueError as error:
        assert "sealed" in str(error)
    else:
        raise AssertionError("training loader opened the sealed holdout")


def test_scope_threshold_maximizes_recall_under_false_accept_constraint() -> None:
    scores = np.asarray([0.95, 0.80, 0.70, 0.40, 0.20])
    supported = np.asarray([True, True, True, False, False])
    threshold, sweep = select_scope_threshold(scores, supported, maximum_false_accept_rate=0.0)
    assert threshold == 0.70
    selected = next(row for row in sweep if row["threshold"] == threshold)
    assert selected["unsupported_false_accept_rate"] == 0.0
    assert selected["in_scope_false_reject_rate"] == 0.0

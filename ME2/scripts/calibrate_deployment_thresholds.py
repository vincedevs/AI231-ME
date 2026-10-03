"""Calibrate VCM scope and intent thresholds from validation predictions only.

The script never opens test or holdout data. It evaluates the actual two-gate
deployment policy: a clip executes only when its in-scope score and calibrated
intent confidence both meet their respective thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate(
    records: list[dict[str, Any]], scope_threshold: float, intent_threshold: float
) -> dict[str, Any]:
    selected = [
        row
        for row in records
        if float(row["in_scope_score"]) >= scope_threshold
        and float(row["intent_confidence"]) >= intent_threshold
    ]
    supported = [row for row in records if bool(row["supported"])]
    unsupported = [row for row in records if not bool(row["supported"])]
    correctly_executed = [
        row
        for row in selected
        if bool(row["supported"]) and row["predicted_intent"] == row["target_intent"]
    ]
    wrong_supported = [
        row
        for row in selected
        if bool(row["supported"]) and row["predicted_intent"] != row["target_intent"]
    ]
    oos_executed = [row for row in selected if not bool(row["supported"])]
    executed = len(selected)
    correct = len(correctly_executed)
    return {
        "scope_threshold": scope_threshold,
        "intent_threshold": intent_threshold,
        "validation_records": len(records),
        "supported_records": len(supported),
        "unsupported_records": len(unsupported),
        "executed": executed,
        "correctly_executed": correct,
        "wrong_supported_executed": len(wrong_supported),
        "unsupported_executed": len(oos_executed),
        "action_precision": correct / executed if executed else 1.0,
        "correct_supported_coverage": correct / len(supported) if supported else 0.0,
        "wrong_supported_execution_rate": (
            len(wrong_supported) / len(supported) if supported else 0.0
        ),
        "unsupported_execution_rate": (
            len(oos_executed) / len(unsupported) if unsupported else 0.0
        ),
    }


def choose_thresholds(
    records: list[dict[str, Any]], precision_target: float, oos_limit: float
) -> dict[str, Any]:
    """Maximize correctly executed supported commands under explicit constraints."""
    scope_scores = np.asarray([float(row["in_scope_score"]) for row in records])
    intent_scores = np.asarray([float(row["intent_confidence"]) for row in records])
    supported = np.asarray([bool(row["supported"]) for row in records])
    correct = np.asarray(
        [
            bool(row["supported"]) and row["predicted_intent"] == row["target_intent"]
            for row in records
        ]
    )
    unsupported_total = int((~supported).sum())
    supported_total = int(supported.sum())
    best: dict[str, Any] | None = None
    for scope_threshold in np.unique(scope_scores):
        indices = np.flatnonzero(scope_scores >= scope_threshold)
        # Descending confidence prefixes exactly represent every possible
        # ``intent_confidence >= threshold`` decision, including tied scores.
        ordered = indices[np.argsort(-intent_scores[indices], kind="stable")]
        ordered_confidence = intent_scores[ordered]
        group_end = np.r_[
            ordered_confidence[:-1] != ordered_confidence[1:],
            True,
        ]
        cumulative_correct = np.cumsum(correct[ordered])
        cumulative_supported = np.cumsum(supported[ordered])
        cumulative_oos = np.cumsum(~supported[ordered])
        for position in np.flatnonzero(group_end):
            executed = int(position + 1)
            correctly_executed = int(cumulative_correct[position])
            oos_executed = int(cumulative_oos[position])
            wrong_supported = int(cumulative_supported[position] - correctly_executed)
            action_precision = correctly_executed / executed
            oos_rate = oos_executed / unsupported_total
            if action_precision < precision_target:
                continue
            if oos_rate > oos_limit:
                continue
            score = (
                correctly_executed,
                action_precision,
                -oos_executed,
                -wrong_supported,
            )
            if best is None or score > tuple(best["selection_score"]):
                best = {
                    "scope_threshold": float(scope_threshold),
                    "intent_threshold": float(ordered_confidence[position]),
                    "validation_records": len(records),
                    "supported_records": supported_total,
                    "unsupported_records": unsupported_total,
                    "executed": executed,
                    "correctly_executed": correctly_executed,
                    "wrong_supported_executed": wrong_supported,
                    "unsupported_executed": oos_executed,
                    "action_precision": action_precision,
                    "correct_supported_coverage": correctly_executed / supported_total,
                    "wrong_supported_execution_rate": wrong_supported / supported_total,
                    "unsupported_execution_rate": oos_rate,
                    "selection_score": list(score),
                }
    if best is None:
        raise ValueError("No threshold pair satisfies the requested validation constraints")
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision-target", type=float, default=0.95)
    parser.add_argument("--oos-limit", type=float, default=0.05)
    parser.add_argument("--current-scope", type=float)
    parser.add_argument("--current-intent", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.precision_target <= 1 or not 0 <= args.oos_limit <= 1:
        raise ValueError("precision target and OOS limit must be probabilities")
    records = read_jsonl(args.predictions)
    if "validation" not in args.predictions.name:
        raise ValueError(
            "Threshold calibration accepts only a validation prediction file; "
            "test and holdout predictions must remain evaluation-only"
        )
    if not records:
        raise ValueError("Validation prediction file is empty")
    required = {
        "supported",
        "target_intent",
        "predicted_intent",
        "in_scope_score",
        "intent_confidence",
    }
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"Validation predictions are missing fields: {sorted(missing)}")
    chosen = choose_thresholds(records, args.precision_target, args.oos_limit)
    report: dict[str, Any] = {
        "source": str(args.predictions),
        "selection_split": "validation_only",
        "test_or_holdout_loaded": False,
        "policy": {
            "execute_when": "in_scope_score >= scope_threshold and intent_confidence >= intent_threshold",
            "precision_target": args.precision_target,
            "unsupported_execution_rate_limit": args.oos_limit,
            "objective": "maximize_correctly_executed_supported_commands",
        },
        "recommended": chosen,
    }
    if args.current_scope is not None and args.current_intent is not None:
        report["current_policy"] = evaluate(records, args.current_scope, args.current_intent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

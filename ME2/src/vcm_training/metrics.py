from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from dataset_pipeline.common import normalize_slot
from dataset_pipeline.schema import INTENTS, SLOT_BY_INTENT

from .models import SLOT_TOKENS


def greedy_ctc_decode(logits: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    predictions = logits.argmax(dim=-1).cpu()
    decoded = []
    for sequence, length in zip(predictions, lengths.cpu(), strict=True):
        characters = []
        previous = -1
        for token in sequence[: int(length)]:
            index = int(token)
            if index != 0 and index != previous:
                characters.append(SLOT_TOKENS[index])
            previous = index
        decoded.append("".join(characters).strip())
    return decoded


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Fit one positive temperature on validation NLL."""
    # Prediction collection uses inference_mode; cloning produces ordinary
    # tensors that autograd may safely save while optimizing temperature.
    logits = logits.clone()
    targets = targets.clone()
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in pairwise(boundaries):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            accuracy = (predictions[selected] == targets[selected]).mean()
            error += selected.mean() * abs(float(accuracy) - float(confidence[selected].mean()))
    return float(error)


def intent_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    calibration_bins: int,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=np.arange(len(INTENTS)),
        zero_division=0,
    )
    averages = {}
    for average in ("macro", "micro", "weighted"):
        p, r, score, _ = precision_recall_fscore_support(
            targets, predictions, average=average, zero_division=0
        )
        averages[average] = {"precision": float(p), "recall": float(r), "f1": float(score)}
    one_hot = np.eye(len(INTENTS), dtype=np.float64)[targets]
    clipped = np.clip(probabilities, 1e-9, 1.0)
    result = {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        **averages,
        "per_intent": {
            intent: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, intent in enumerate(INTENTS)
        },
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=np.arange(len(INTENTS))
        ).tolist(),
        "ece": expected_calibration_error(probabilities, targets, calibration_bins),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "nll": float(-np.mean(np.log(clipped[np.arange(targets.size), targets]))),
    }
    if bootstrap_samples > 0:
        result["bootstrap_confidence_intervals"] = bootstrap_intent_intervals(
            predictions,
            targets,
            bootstrap_samples,
            bootstrap_confidence,
            bootstrap_seed,
        )
    return result


def bootstrap_intent_intervals(
    predictions: np.ndarray,
    targets: np.ndarray,
    samples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any]:
    """Estimate test-set uncertainty without assuming normally distributed scores."""
    generator = np.random.default_rng(seed)
    accuracy_values = np.empty(samples, dtype=np.float64)
    macro_f1_values = np.empty(samples, dtype=np.float64)
    classes = len(INTENTS)
    for index in range(samples):
        selected = generator.integers(0, targets.size, size=targets.size)
        actual = targets[selected]
        predicted = predictions[selected]
        matrix = np.bincount(actual * classes + predicted, minlength=classes * classes).reshape(
            classes, classes
        )
        true_positive = np.diag(matrix).astype(np.float64)
        false_positive = matrix.sum(axis=0) - true_positive
        false_negative = matrix.sum(axis=1) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        per_class_f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator > 0,
        )
        accuracy_values[index] = true_positive.sum() / max(matrix.sum(), 1)
        macro_f1_values[index] = per_class_f1.mean()
    alpha = (1.0 - confidence) / 2.0

    def interval(values: np.ndarray) -> dict[str, float]:
        lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
        return {"lower": float(lower), "upper": float(upper)}

    return {
        "method": "nonparametric_bootstrap",
        "confidence": confidence,
        "samples": samples,
        "seed": seed,
        "accuracy": interval(accuracy_values),
        "macro_f1": interval(macro_f1_values),
    }


def scope_threshold_sweep(scores: np.ndarray, supported: np.ndarray) -> list[dict[str, float]]:
    """Enumerate validation rejection trade-offs without consulting test data."""
    candidates = np.unique(np.concatenate(([0.0, 1.0], scores.astype(np.float64))))
    rows = []
    for threshold in candidates:
        accepted = scores >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "unsupported_false_accept_rate": float(
                    (accepted & ~supported).sum() / max((~supported).sum(), 1)
                ),
                "in_scope_false_reject_rate": float(
                    ((~accepted) & supported).sum() / max(supported.sum(), 1)
                ),
            }
        )
    return rows


def select_scope_threshold(
    scores: np.ndarray,
    supported: np.ndarray,
    maximum_false_accept_rate: float = 0.05,
) -> tuple[float, list[dict[str, float]]]:
    """Maximize supported recall subject to a pre-registered OOS FAR limit."""
    sweep = scope_threshold_sweep(scores, supported)
    feasible = [
        row
        for row in sweep
        if row["unsupported_false_accept_rate"] <= maximum_false_accept_rate
    ]
    candidates = feasible or sweep
    selected = min(
        candidates,
        key=lambda row: (
            row["in_scope_false_reject_rate"],
            row["unsupported_false_accept_rate"],
            -row["threshold"],
        ),
    )
    return float(selected["threshold"]), sweep


def scope_metrics(
    scores: np.ndarray,
    supported: np.ndarray,
    threshold: float,
    ood_types: Sequence[str],
    category_partitions: Sequence[str],
) -> dict[str, Any]:
    accepted = scores >= threshold
    false_accept = accepted & ~supported
    false_reject = ~accepted & supported
    has_supported = bool(supported.any())
    has_unsupported = bool((~supported).any())
    if has_supported and has_unsupported:
        false_positive_rate, true_positive_rate, _ = roc_curve(supported, scores)
        candidates = np.flatnonzero(true_positive_rate >= 0.95)
        fpr_at_95 = float(false_positive_rate[candidates[0]]) if candidates.size else 1.0
        auroc = float(roc_auc_score(supported, scores))
        auprc = float(average_precision_score(supported, scores))
    else:
        fpr_at_95 = None
        auroc = None
        auprc = None
    breakdown = {}
    for label, values in (("ood_type", ood_types), ("category_partition", category_partitions)):
        breakdown[label] = {}
        for value in sorted(set(values)):
            selected = np.array([item == value for item in values]) & ~supported
            if selected.any():
                breakdown[label][value] = {
                    "records": int(selected.sum()),
                    "false_accept_rate": float(accepted[selected].mean()),
                    "mean_in_scope_score": float(scores[selected].mean()),
                }
    return {
        "threshold": float(threshold),
        "auroc": auroc,
        "auprc": auprc,
        "fpr_at_95_percent_in_scope_recall": fpr_at_95,
        "unsupported_false_accept_rate": float(false_accept.sum() / max((~supported).sum(), 1)),
        "in_scope_false_reject_rate": float(false_reject.sum() / max(supported.sum(), 1)),
        "coverage": float(accepted.mean()),
        "breakdown": breakdown,
    }


def normalized_slot_identity(intent_index: int, text: str) -> Any:
    slot_name = SLOT_BY_INTENT.get(INTENTS[intent_index])
    if slot_name is None:
        return None
    normalized = normalize_slot(slot_name, text)
    if "value" in normalized:
        return (normalized.get("value"), normalized.get("unit"))
    if "seconds" in normalized:
        return normalized["seconds"]
    return normalized.get("text", "")


def slot_and_joint_metrics(
    predicted_intents: np.ndarray,
    target_intents: np.ndarray,
    predicted_slots: Sequence[str],
    target_slots: Sequence[str],
) -> dict[str, Any]:
    strict_true_positive = strict_predicted = strict_expected = 0
    normalized_true_positive = normalized_predicted = normalized_expected = 0
    joint = []
    edit_distance_total = 0
    target_character_total = 0
    for predicted_intent, target_intent, predicted, target in zip(
        predicted_intents, target_intents, predicted_slots, target_slots, strict=True
    ):
        slot_name = SLOT_BY_INTENT.get(INTENTS[int(target_intent)])
        intent_correct = int(predicted_intent) == int(target_intent)
        if slot_name is None:
            joint.append(intent_correct)
            continue
        strict_expected += 1
        normalized_expected += 1
        if predicted:
            strict_predicted += 1
            normalized_predicted += 1
        strict_match = predicted == target
        normalized_match = normalized_slot_identity(
            int(target_intent), predicted
        ) == normalized_slot_identity(int(target_intent), target)
        strict_true_positive += int(strict_match)
        normalized_true_positive += int(normalized_match)
        joint.append(intent_correct and normalized_match)
        edit_distance_total += levenshtein(predicted, target)
        target_character_total += len(target)

    def scores(true_positive: int, predicted: int, expected: int) -> dict[str, float]:
        precision = true_positive / max(predicted, 1)
        recall = true_positive / max(expected, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {"precision": precision, "recall": recall, "f1": f1}

    return {
        "strict": scores(strict_true_positive, strict_predicted, strict_expected),
        "normalized": scores(normalized_true_positive, normalized_predicted, normalized_expected),
        "character_error_rate": edit_distance_total / max(target_character_total, 1),
        "joint_intent_slot_accuracy": float(np.mean(joint)) if joint else math.nan,
        "slotted_records": strict_expected,
    }


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]

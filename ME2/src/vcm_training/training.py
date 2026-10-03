from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_pipeline.schema import INTENTS

from .data import CommandDataset
from .metrics import (
    fit_temperature,
    greedy_ctc_decode,
    intent_metrics,
    scope_metrics,
    select_scope_threshold,
    slot_and_joint_metrics,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _nonfinite_float_paths(value: Any, path: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, float) and not math.isfinite(value):
        paths.append(path)
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            paths.extend(_nonfinite_float_paths(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_nonfinite_float_paths(item, f"{path}[{index}]"))
    return paths


def repair_legacy_binary_scope_history(
    history: list[dict[str, Any]], rejection_strategy: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Repair only the known pre-fix unsupported-only minibatch log entries.

    Before the explicit empty-batch guard, binary-scope training correctly had
    no intent target for an unsupported-only minibatch. PyTorch returned NaN
    for that empty mean even though its gradient contribution was zero. The
    model and validation metrics remained finite, but the epoch-level intent
    and total loss aggregates became unavailable. Preserve that missingness as
    JSON null and reject every other non-finite checkpoint pattern.
    """
    repaired = copy.deepcopy(history)
    repairs = []
    expected_paths = {"training.intent_loss", "training.total_loss"}
    for index, row in enumerate(repaired):
        paths = set(_nonfinite_float_paths(row))
        if not paths:
            continue
        if rejection_strategy != "binary_scope" or paths != expected_paths:
            formatted = ", ".join(sorted(paths))
            raise FloatingPointError(
                f"Checkpoint history contains unexpected non-finite values: {formatted}"
            )
        row["training"]["intent_loss"] = None
        row["training"]["total_loss"] = None
        repairs.append(
            {
                "history_index": index,
                "epoch": row.get("epoch"),
                "fields": sorted(expected_paths),
                "replacement": None,
                "reason": "legacy unsupported-only minibatch had no intent target",
            }
        )
    return repaired, repairs


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = dict(batch)
    for key in (
        "waveform",
        "lengths",
        "intent_targets",
        "scope_targets",
        "slot_targets",
        "slot_target_lengths",
    ):
        moved[key] = batch[key].to(device, non_blocking=True)
    return moved


def compute_loss(
    model: torch.nn.Module,
    batch: dict[str, Any],
    rejection_strategy: str,
    training_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    intent_logits, slot_logits, scope_logits, sequence_lengths = model(
        batch["waveform"], batch["lengths"]
    )
    supported = batch["scope_targets"].bool()
    if rejection_strategy == "unknown_class":
        intent_targets = batch["intent_targets"].clone()
        intent_targets[~supported] = len(INTENTS)
        intent_loss = F.cross_entropy(
            intent_logits,
            intent_targets,
            label_smoothing=float(training_config["label_smoothing"]),
        )
    elif supported.any():
        intent_loss = F.cross_entropy(
            intent_logits[supported],
            batch["intent_targets"][supported],
            label_smoothing=float(training_config["label_smoothing"]),
        )
    else:
        # Binary-scope batches may contain only unsupported commands. They
        # provide a valid scope target but deliberately have no intent target.
        intent_loss = intent_logits.sum() * 0.0

    slotted_indices = [
        index
        for index, sequence in enumerate(batch["slot_target_sequences"])
        if sequence and bool(supported[index])
    ]
    if slotted_indices:
        selected = torch.tensor(slotted_indices, device=slot_logits.device)
        targets = torch.tensor(
            [token for index in slotted_indices for token in batch["slot_target_sequences"][index]],
            dtype=torch.long,
            device=slot_logits.device,
        )
        target_lengths = torch.tensor(
            [len(batch["slot_target_sequences"][index]) for index in slotted_indices],
            dtype=torch.long,
            device=slot_logits.device,
        )
        slot_loss = F.ctc_loss(
            slot_logits.index_select(0, selected).log_softmax(dim=-1).transpose(0, 1),
            targets,
            sequence_lengths.index_select(0, selected),
            target_lengths,
            blank=0,
            zero_infinity=True,
        )
    else:
        slot_loss = slot_logits.sum() * 0.0

    if rejection_strategy == "binary_scope":
        scope_loss = F.binary_cross_entropy_with_logits(
            scope_logits.squeeze(1), batch["scope_targets"]
        )
    else:
        scope_loss = scope_logits.sum() * 0.0
    total = (
        intent_loss
        + float(training_config["slot_loss_weight"]) * slot_loss
        + float(training_config["scope_loss_weight"]) * scope_loss
    )
    if not bool(torch.isfinite(total)):
        supported_count = int(supported.sum())
        raise FloatingPointError(
            "Non-finite training loss before backward pass: "
            f"strategy={rejection_strategy}, batch={supported.numel()}, "
            f"supported={supported_count}, unsupported={supported.numel() - supported_count}"
        )
    return total, {
        "intent_loss": float(intent_loss.detach()),
        "slot_loss": float(slot_loss.detach()),
        "scope_loss": float(scope_loss.detach()),
        "total_loss": float(total.detach()),
    }


def scope_scores(
    intent_logits: torch.Tensor,
    scope_logits: torch.Tensor,
    rejection_strategy: str,
    temperature: float = 1.0,
) -> torch.Tensor:
    if rejection_strategy == "confidence":
        return torch.softmax(intent_logits[:, : len(INTENTS)] / temperature, dim=1).amax(dim=1)
    if rejection_strategy == "unknown_class":
        return 1.0 - torch.softmax(intent_logits / temperature, dim=1)[:, len(INTENTS)]
    return torch.sigmoid(scope_logits.squeeze(1))


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    rejection_strategy: str,
    description: str,
) -> dict[str, Any]:
    model.eval()
    supported_logits = []
    supported_targets = []
    supported_slots: list[str] = []
    supported_slot_targets: list[str] = []
    all_intent_logits = []
    all_scope_logits = []
    all_supported = []
    sample_ids: list[str] = []
    ood_types: list[str] = []
    category_partitions: list[str] = []
    sources: list[str] = []
    for batch in tqdm(loader, desc=description, unit="batch", leave=False):
        batch = move_batch(batch, device)
        outputs = model(batch["waveform"], batch["lengths"])
        intent_logits, slot_logits, scope_logits, sequence_lengths = outputs
        supported = batch["scope_targets"].bool()
        all_intent_logits.append(intent_logits.cpu())
        all_scope_logits.append(scope_logits.cpu())
        all_supported.append(supported.cpu())
        decoded = greedy_ctc_decode(slot_logits, sequence_lengths)
        if supported.any():
            supported_logits.append(intent_logits[supported, : len(INTENTS)].cpu())
            supported_targets.append(batch["intent_targets"][supported].cpu())
            selected = supported.cpu().tolist()
            supported_slots.extend(value for value, keep in zip(decoded, selected) if keep)
            supported_slot_targets.extend(
                value for value, keep in zip(batch["slot_texts"], selected) if keep
            )
        sample_ids.extend(batch["sample_ids"])
        ood_types.extend(batch["ood_types"])
        category_partitions.extend(batch["category_partitions"])
        sources.extend(batch["source_datasets"])
    return {
        "supported_logits": torch.cat(supported_logits),
        "supported_targets": torch.cat(supported_targets),
        "supported_predicted_slots": supported_slots,
        "supported_target_slots": supported_slot_targets,
        "all_intent_logits": torch.cat(all_intent_logits),
        "all_scope_logits": torch.cat(all_scope_logits),
        "all_supported": torch.cat(all_supported),
        "sample_ids": sample_ids,
        "ood_types": ood_types,
        "category_partitions": category_partitions,
        "sources": sources,
    }


def evaluate_collected(
    collected: dict[str, Any],
    rejection_strategy: str,
    evaluation_config: dict[str, Any],
    temperature: float,
    threshold: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    logits = collected["supported_logits"] / temperature
    probabilities = torch.softmax(logits, dim=1).numpy()
    targets = collected["supported_targets"].numpy()
    predicted_intents = probabilities.argmax(axis=1)
    intent = intent_metrics(
        probabilities,
        targets,
        int(evaluation_config["calibration_bins"]),
        int(evaluation_config["bootstrap_samples"]),
        float(evaluation_config["bootstrap_confidence"]),
        int(evaluation_config["bootstrap_seed"]),
    )
    slots = slot_and_joint_metrics(
        predicted_intents,
        targets,
        collected["supported_predicted_slots"],
        collected["supported_target_slots"],
    )
    scores = scope_scores(
        collected["all_intent_logits"],
        collected["all_scope_logits"],
        rejection_strategy,
        temperature,
    ).numpy()
    supported = collected["all_supported"].numpy().astype(bool)
    if threshold is None:
        selected_threshold, threshold_sweep = select_scope_threshold(
            scores,
            supported,
            float(evaluation_config.get("maximum_unsupported_false_accept_rate", 0.05)),
        )
    else:
        selected_threshold = threshold
        threshold_sweep = []
    scope = scope_metrics(
        scores,
        supported,
        selected_threshold,
        collected["ood_types"],
        collected["category_partitions"],
    )

    all_logits = collected["all_intent_logits"][:, : len(INTENTS)] / temperature
    all_probabilities = torch.softmax(all_logits, dim=1).numpy()
    all_predictions = all_probabilities.argmax(axis=1)
    all_confidence = all_probabilities.max(axis=1)
    accepted = scores >= selected_threshold
    supported_cursor = 0
    rows = []
    for index, sample_id in enumerate(collected["sample_ids"]):
        row = {
            "sample_id": sample_id,
            "source_dataset": collected["sources"][index],
            "supported": bool(supported[index]),
            "accepted": bool(accepted[index]),
            "in_scope_score": float(scores[index]),
            "predicted_intent": INTENTS[int(all_predictions[index])],
            "intent_confidence": float(all_confidence[index]),
            "ood_type": collected["ood_types"][index],
            "category_partition": collected["category_partitions"][index],
        }
        if supported[index]:
            target_index = int(targets[supported_cursor])
            row.update(
                {
                    "target_intent": INTENTS[target_index],
                    "predicted_slot": collected["supported_predicted_slots"][supported_cursor],
                    "target_slot": collected["supported_target_slots"][supported_cursor],
                }
            )
            supported_cursor += 1
        rows.append(row)
    accepted_supported = accepted & supported
    accepted_accuracy = (
        float((all_predictions[accepted_supported] == targets[accepted[supported]]).mean())
        if accepted_supported.any()
        else None
    )
    return (
        {
            "intent": intent,
            "slots": slots,
            "scope": scope,
            "threshold_sweep": threshold_sweep,
            "accepted_supported_intent_accuracy": accepted_accuracy,
            "temperature": float(temperature),
        },
        rows,
        selected_threshold,
    )


def train_model(
    model: torch.nn.Module,
    train_dataset: CommandDataset,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    rejection_strategy: str,
    training_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    output_directory: Path,
    experiment_identity: dict[str, Any],
    epoch_callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], float, float]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(training_config["epochs"]),
        eta_min=float(training_config["minimum_learning_rate"]),
    )
    use_amp = bool(training_config["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint = output_directory / "last.pt"
    start_epoch = 0
    best_score = -math.inf
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state["best_score"])
        stale_epochs = int(state["stale_epochs"])
        history, history_repairs = repair_legacy_binary_scope_history(
            list(state["history"]), rejection_strategy
        )
        if history_repairs:
            atomic_json(
                output_directory / "history_recovery.json",
                {
                    **experiment_identity,
                    "checkpoint": str(checkpoint),
                    "repairs": history_repairs,
                },
            )

    configured_epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    # A run can fail after training, for example during ONNX validation. When
    # its checkpoint already reached the epoch limit or early-stopping
    # patience, recovery must not silently add another optimization epoch.
    final_epoch = (
        start_epoch
        if start_epoch >= configured_epochs or stale_epochs >= patience
        else configured_epochs
    )

    tensorboard = SummaryWriter(
        log_dir=str(output_directory / "tensorboard"), purge_step=start_epoch
    )
    started = time.time()
    for epoch in range(start_epoch, final_epoch):
        train_dataset.set_epoch(epoch)
        sampler_generator = getattr(train_loader.sampler, "generator", None)
        if sampler_generator is not None:
            sampler_generator.manual_seed(int(experiment_identity["seed"]) + epoch)
        model.train()
        running = {key: 0.0 for key in ("total_loss", "intent_loss", "slot_loss", "scope_loss")}
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}", unit="batch")
        for batch_index, batch in enumerate(progress):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss, components = compute_loss(model, batch, rejection_strategy, training_config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            for key in running:
                running[key] += components[key]
            progress.set_postfix(loss=f"{running['total_loss'] / (batch_index + 1):.4f}")
            if batch_index % 50 == 0:
                atomic_json(
                    output_directory / "status.json",
                    {
                        **experiment_identity,
                        "state": "training",
                        "epoch": epoch + 1,
                        "epochs": configured_epochs,
                        "batch": batch_index + 1,
                        "batches": len(train_loader),
                        "mean_loss": running["total_loss"] / (batch_index + 1),
                        "best_validation_macro_f1": (
                            best_score if math.isfinite(best_score) else None
                        ),
                        "elapsed_seconds": time.time() - started,
                    },
                )
        scheduler.step()
        validation = collect_predictions(
            model, validation_loader, device, rejection_strategy, "Validation"
        )
        validation_temperature = fit_temperature(
            validation["supported_logits"], validation["supported_targets"]
        )
        validation_metrics, _, threshold = evaluate_collected(
            validation,
            rejection_strategy,
            {**evaluation_config, "bootstrap_samples": 0},
            validation_temperature,
            threshold=None,
        )
        score = float(validation_metrics["intent"]["macro"]["f1"])
        epoch_metrics = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "training": {key: value / max(len(train_loader), 1) for key, value in running.items()},
            "validation": validation_metrics,
        }
        history.append(epoch_metrics)
        for key, value in epoch_metrics["training"].items():
            tensorboard.add_scalar(f"training/{key}", value, epoch + 1)
        tensorboard.add_scalar("validation/intent_macro_f1", score, epoch + 1)
        tensorboard.add_scalar(
            "validation/joint_intent_slot_accuracy",
            validation_metrics["slots"]["joint_intent_slot_accuracy"],
            epoch + 1,
        )
        tensorboard.add_scalar(
            "validation/unsupported_auroc",
            validation_metrics["scope"]["auroc"],
            epoch + 1,
        )
        tensorboard.add_scalar(
            "optimization/learning_rate", optimizer.param_groups[0]["lr"], epoch + 1
        )
        improved = score > best_score
        if improved:
            best_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "stale_epochs": stale_epochs,
            "history": history,
            "temperature": validation_temperature,
            "scope_threshold": threshold,
            "experiment_identity": experiment_identity,
        }
        torch.save(state, checkpoint)
        if improved:
            torch.save(state, output_directory / "best.pt")
        atomic_json(output_directory / "history.json", history)
        if epoch_callback is not None:
            epoch_callback(epoch + 1, validation_metrics)
        if stale_epochs >= patience:
            break

    tensorboard.close()
    best = torch.load(output_directory / "best.pt", map_location=device, weights_only=False)
    best["history"], _ = repair_legacy_binary_scope_history(
        list(best["history"]), rejection_strategy
    )
    model.load_state_dict(best["model"])
    return best, float(best["temperature"]), float(best["scope_threshold"])

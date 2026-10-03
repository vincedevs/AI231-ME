from __future__ import annotations

import csv
import json
import platform
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader

from dataset_pipeline.common import sha256_file, stable_digest, write_jsonl
from dataset_pipeline.schema import INTENTS, SLOT_BY_INTENT

from .data import (
    CommandDataset,
    balanced_sampler,
    collate_commands,
    dataset_summary,
    load_experiment_records,
    resolve_audio_path,
)
from .export import build_deployment_metadata, export_onnx
from .models import SLOT_TOKENS, build_model
from .training import (
    atomic_json,
    collect_predictions,
    evaluate_collected,
    seed_everything,
    train_model,
)


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def seed_worker(worker_id: int) -> None:
    np.random.seed(torch.initial_seed() % 2**32)


def make_loader(
    dataset: CommandDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    sampler: Any = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_commands,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )


def select_smoke_records(records: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        label = str(row["canonical_intent"] if row["supported"] else "UNSUPPORTED")
        groups.setdefault(label, []).append(row)
    selected = []
    while len(selected) < maximum and any(groups.values()):
        for label in sorted(groups):
            if groups[label] and len(selected) < maximum:
                selected.append(groups[label].pop(0))
    return selected


def validate_release(prepared_root: Path) -> dict[str, Any]:
    release_path = prepared_root / "release.json"
    if not release_path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset is missing. Run scripts/prepare_dataset.py first: {release_path}"
        )
    release = json.loads(release_path.read_text(encoding="utf-8"))
    for relative, expected in release["manifest_sha256"].items():
        path = prepared_root / relative
        if sha256_file(path) != expected:
            raise ValueError(f"Prepared manifest checksum mismatch: {path}")
    if release.get("holdout_policy") != "sealed_for_final_raspberry_pi_evaluation":
        raise ValueError("Prepared release does not declare a sealed holdout")
    return release


def initialize_study_identity(
    output_root: Path,
    config: dict[str, Any],
    release_path: Path,
) -> dict[str, str]:
    """Prevent resumable outputs from crossing dataset or study configurations."""
    identity = {
        "dataset_release_sha256": sha256_file(release_path),
        "experiment_config_sha256": stable_digest(json.dumps(config, sort_keys=True)),
    }
    path = output_root / "study_identity.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity:
            raise ValueError(
                "Experiment output belongs to a different dataset release or study "
                f"configuration: {output_root}. Use a clean output directory."
            )
        return identity
    existing_artifacts = [item for item in output_root.iterdir() if item.name != ".DS_Store"]
    if existing_artifacts:
        raise ValueError(
            "Experiment output predates the study-identity guard and cannot be safely "
            f"resumed: {output_root}. Archive it and use a clean output directory."
        )
    atomic_json(path, identity)
    return identity


def validate_audio_paths(prepared_root: Path, records: list[dict[str, Any]]) -> None:
    missing = []
    for row in records:
        path = resolve_audio_path(prepared_root, row)
        if not path.is_file():
            missing.append(str(path))
            if len(missing) == 20:
                break
    if missing:
        raise FileNotFoundError(f"Missing prepared audio files: {missing}")


def run_onnx_preflight(
    output_root: Path,
    config: dict[str, Any],
    architectures: list[str],
) -> None:
    directory = output_root / "onnx_preflight"
    directory.mkdir(parents=True, exist_ok=True)
    evaluation = {
        **config["evaluation"],
        "latency_warmup_runs": 1,
        "latency_timed_runs": 1,
    }
    for architecture in architectures:
        report_path = directory / f"{architecture}.json"
        if report_path.is_file():
            continue
        model = build_model(architecture, "binary_scope", config["audio"])
        model_path = directory / f"{architecture}.onnx"
        report = export_onnx(model, model_path, config["audio"], evaluation)
        atomic_json(report_path, report)
        model_path.unlink()


def save_plots(metrics: dict[str, Any], history: list[dict[str, Any]], output: Path) -> None:
    confusion = np.asarray(metrics["intent"]["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(range(len(INTENTS)), INTENTS, rotation=90, fontsize=7)
    axis.set_yticks(range(len(INTENTS)), INTENTS, fontsize=7)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    epochs = [row["epoch"] for row in history]
    losses = [row["training"]["total_loss"] for row in history]
    scores = [row["validation"]["intent"]["macro"]["f1"] for row in history]
    figure, left = plt.subplots(figsize=(8, 5))
    left.plot(epochs, losses, color="tab:blue")
    left.set_xlabel("Epoch")
    left.set_ylabel("Training loss", color="tab:blue")
    right = left.twinx()
    right.plot(epochs, scores, color="tab:orange")
    right.set_ylabel("Validation macro F1", color="tab:orange")
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=180)
    plt.close(figure)


def training_records(
    prepared_root: Path,
    condition: str,
    rejection: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        split: load_experiment_records(prepared_root, split, condition, rejection)
        for split in ("train", "validation")
    }


def trial_score(metrics: dict[str, Any], config: dict[str, Any]) -> float:
    selection = config["selection"]
    intent_f1 = float(metrics["intent"]["macro"]["f1"])
    joint = float(metrics["slots"]["joint_intent_slot_accuracy"])
    far = float(metrics["scope"]["unsupported_false_accept_rate"])
    frr = float(metrics["scope"]["in_scope_false_reject_rate"])
    feasible = (
        far <= float(selection["maximum_unsupported_false_accept_rate"])
        and frr <= float(selection["maximum_in_scope_false_reject_rate"])
    )
    return (1.0 if feasible else 0.0) + intent_f1 + 0.1 * joint - far - 0.25 * frr


def run_training_job(
    project_root: Path,
    config: dict[str, Any],
    device: torch.device,
    *,
    architecture: str,
    data_condition: str,
    rejection_strategy: str,
    seed: int,
    output: Path,
    training_overrides: dict[str, Any] | None = None,
    smoke_test: bool = False,
    epoch_callback=None,
    export_model: bool = True,
) -> dict[str, Any]:
    prepared_root = resolve(project_root, config["paths"]["prepared_dataset"])
    current_release_sha256 = sha256_file(prepared_root / "release.json")
    metrics_path = output / "metrics.json"
    if metrics_path.is_file():
        resolved_path = output / "config.json"
        if not resolved_path.is_file():
            raise ValueError(f"Completed run is missing its resolved configuration: {output}")
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        if resolved.get("dataset_release_sha256") != current_release_sha256:
            raise ValueError(
                "Refusing to reuse a completed run from another dataset release: "
                f"{output}"
            )
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "architecture": architecture,
        "data_condition": data_condition,
        "rejection_strategy": rejection_strategy,
        "seed": int(seed),
    }
    atomic_json(output / "status.json", {**identity, "state": "preparing"})
    seed_everything(seed)
    records = training_records(prepared_root, data_condition, rejection_strategy)
    if smoke_test:
        records = {
            "train": select_smoke_records(records["train"], 96),
            "validation": select_smoke_records(records["validation"], 64),
        }
    validate_audio_paths(prepared_root, records["train"] + records["validation"])
    summaries = {split: dataset_summary(rows) for split, rows in records.items()}
    atomic_json(output / "dataset_summary.json", summaries)
    training_config = {**config["training"], **(training_overrides or {})}
    if smoke_test:
        training_config.update(
            {
                "epochs": 1,
                "early_stopping_patience": 1,
                "batch_size": 8,
                "num_workers": 0,
                "mixed_precision": False,
            }
        )
    train_dataset = CommandDataset(
        records["train"], prepared_root, config["audio"], config["augmentation"], seed
    )
    validation_dataset = CommandDataset(
        records["validation"], prepared_root, config["audio"], None, seed
    )
    sampler = balanced_sampler(
        records["train"], float(training_config["unsupported_batch_fraction"]), seed
    )
    train_loader = make_loader(
        train_dataset,
        int(training_config["batch_size"]),
        int(training_config["num_workers"]),
        device,
        sampler,
    )
    validation_loader = make_loader(
        validation_dataset,
        int(training_config["batch_size"]),
        int(training_config["num_workers"]),
        device,
    )
    model = build_model(architecture, rejection_strategy, config["audio"]).to(device)
    resolved_config = {
        **config,
        "training": training_config,
        "experiment": identity,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "device": str(device),
        },
        "dataset_release_sha256": current_release_sha256,
    }
    atomic_json(output / "config.json", resolved_config)
    best, temperature, threshold = train_model(
        model,
        train_dataset,
        train_loader,
        validation_loader,
        device,
        rejection_strategy,
        training_config,
        config["evaluation"],
        output,
        identity,
        epoch_callback=epoch_callback,
    )
    collected = collect_predictions(
        model, validation_loader, device, rejection_strategy, "Locked validation"
    )
    validation_metrics, validation_rows, _ = evaluate_collected(
        collected,
        rejection_strategy,
        config["evaluation"],
        temperature,
        threshold,
    )
    write_jsonl(output / "validation_predictions.jsonl", validation_rows)
    atomic_json(output / "validation_metrics.json", validation_metrics)
    onnx_report = None
    if export_model:
        onnx_report = export_onnx(
            model, output / "model.onnx", config["audio"], config["evaluation"]
        )
        metadata = build_deployment_metadata(
            audio_config=config["audio"],
            intent_labels=INTENTS,
            slot_by_intent=SLOT_BY_INTENT,
            slot_tokens=SLOT_TOKENS,
            rejection_strategy=rejection_strategy,
            temperature=temperature,
            scope_threshold=threshold,
            minimum_intent_confidence=float(config["evaluation"]["minimum_intent_confidence"]),
            experiment_identity=identity,
        )
        atomic_json(output / "deployment_metadata.json", metadata)
    result = {
        **identity,
        "state": "complete",
        "dataset": summaries,
        "validation": validation_metrics,
        "test": None,
        "onnx": onnx_report,
        "best_epoch": int(best["epoch"]) + 1,
        "best_validation_macro_f1": float(best["best_score"]),
        "training_overrides": training_overrides or {},
    }
    atomic_json(metrics_path, result)
    save_plots(validation_metrics, best["history"], output)
    atomic_json(output / "status.json", {**identity, "state": "complete"})
    return result


def suggest_parameters(trial: optuna.Trial, config: dict[str, Any]) -> dict[str, Any]:
    space = config["tuning"]["search_space"]
    return {
        "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
        "weight_decay": trial.suggest_float("weight_decay", *space["weight_decay"], log=True),
        "label_smoothing": trial.suggest_categorical("label_smoothing", space["label_smoothing"]),
        "slot_loss_weight": trial.suggest_categorical(
            "slot_loss_weight", space["slot_loss_weight"]
        ),
        "scope_loss_weight": trial.suggest_categorical(
            "scope_loss_weight", space["scope_loss_weight"]
        ),
        "unsupported_batch_fraction": trial.suggest_categorical(
            "unsupported_batch_fraction", space["unsupported_batch_fraction"]
        ),
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "epochs": int(config["tuning"]["epochs_per_trial"]),
        "early_stopping_patience": int(config["tuning"]["early_stopping_patience"]),
    }


def tune_architecture(
    project_root: Path,
    config: dict[str, Any],
    device: torch.device,
    architecture: str,
    output_root: Path,
    smoke_test: bool,
) -> dict[str, Any]:
    tuning = config["tuning"]
    directory = output_root / "optimization" / architecture
    directory.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=architecture,
        storage=f"sqlite:///{directory / 'study.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=int(config["seed"]),
            n_startup_trials=int(tuning.get("sampler_startup_trials", 10)),
        ),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=int(tuning["pruning_warmup_epochs"])
        ),
    )

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_parameters(trial, config)

        def callback(epoch: int, metrics: dict[str, Any]) -> None:
            trial.report(trial_score(metrics, config), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = run_training_job(
            project_root,
            config,
            device,
            architecture=architecture,
            data_condition=str(tuning["reference_data_condition"]),
            rejection_strategy=str(tuning["reference_rejection_strategy"]),
            seed=int(config["seed"]),
            output=directory / f"trial_{trial.number:04d}",
            training_overrides=parameters,
            smoke_test=smoke_test,
            epoch_callback=callback,
            export_model=False,
        )
        return trial_score(result["validation"], config)

    requested = 1 if smoke_test else int(tuning["trials_per_architecture"])
    finished = sum(trial.state.is_finished() for trial in study.trials)
    remaining = max(0, requested - finished)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True)
    completed = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError(f"No HPO trial completed successfully for {architecture}")
    best = dict(study.best_trial.params)
    atomic_json(
        directory / "best_hyperparameters.json",
        {
            "architecture": architecture,
            "objective": float(study.best_value),
            "trial": int(study.best_trial.number),
            "parameters": best,
        },
    )
    study.trials_dataframe().to_csv(directory / "trials.csv", index=False)
    return best


def validation_order(result: dict[str, Any], config: dict[str, Any]) -> tuple[Any, ...]:
    metrics = result["validation"]
    scope = metrics["scope"]
    feasible = (
        scope["unsupported_false_accept_rate"]
        <= float(config["selection"]["maximum_unsupported_false_accept_rate"])
        and scope["in_scope_false_reject_rate"]
        <= float(config["selection"]["maximum_in_scope_false_reject_rate"])
    )
    onnx = result.get("onnx") or {}
    latency = onnx.get("cpu_latency_two_second_audio_ms", {}).get("p95", float("inf"))
    return (
        int(feasible),
        float(metrics["intent"]["macro"]["f1"]),
        float(metrics["slots"]["joint_intent_slot_accuracy"]),
        -float(scope["in_scope_false_reject_rate"]),
        -int(onnx.get("size_bytes", 10**18)),
        -float(latency),
    )


def evaluate_test(
    project_root: Path,
    config: dict[str, Any],
    device: torch.device,
    result: dict[str, Any],
    output: Path,
    smoke_test: bool = False,
) -> dict[str, Any]:
    if result.get("test"):
        return result
    prepared_root = resolve(project_root, config["paths"]["prepared_dataset"])
    records = load_experiment_records(
        prepared_root, "test", result["data_condition"], result["rejection_strategy"]
    )
    model = build_model(
        result["architecture"], result["rejection_strategy"], config["audio"]
    ).to(device)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    temperature = float(checkpoint["temperature"])
    threshold = float(checkpoint["scope_threshold"])
    real_records = [row for row in records if not row["is_synthetic"]]
    synthetic_records = [row for row in records if row["is_synthetic"]]
    strata = (
        {
            "all": select_smoke_records(records, 96),
            "real": select_smoke_records(real_records, 64),
            "synthetic": select_smoke_records(synthetic_records, 64),
        }
        if smoke_test
        else {"all": records, "real": real_records, "synthetic": synthetic_records}
    )
    test_metrics = {}
    for name, rows in strata.items():
        dataset = CommandDataset(rows, prepared_root, config["audio"], None, int(result["seed"]))
        loader = make_loader(
            dataset,
            int(config["training"]["batch_size"]),
            0 if smoke_test else int(config["training"]["num_workers"]),
            device,
        )
        collected = collect_predictions(
            model, loader, device, result["rejection_strategy"], f"Test ({name})"
        )
        metrics, predictions, _ = evaluate_collected(
            collected,
            result["rejection_strategy"],
            config["evaluation"],
            temperature,
            threshold,
        )
        test_metrics[name] = metrics
        write_jsonl(output / f"test_predictions_{name}.jsonl", predictions)
    result = {**result, "test": test_metrics}
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "test_metrics.json", test_metrics)
    return result


def run_identity(result: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(result["architecture"]),
        str(result["data_condition"]),
        str(result["rejection_strategy"]),
    )


def run_directory(output_root: Path, result: dict[str, Any]) -> Path:
    return (
        output_root
        / "runs"
        / result["architecture"]
        / result["data_condition"]
        / result["rejection_strategy"]
        / f"seed_{result['seed']}"
    )


def write_comparison(
    output_root: Path,
    results: list[dict[str, Any]],
    stem: str = "comparison",
) -> None:
    rows = []
    for result in results:
        validation = result["validation"]
        real_test = (result.get("test") or {}).get("real")
        rows.append(
            {
                "architecture": result["architecture"],
                "data_condition": result["data_condition"],
                "rejection_strategy": result["rejection_strategy"],
                "seed": result["seed"],
                "validation_intent_macro_f1": validation["intent"]["macro"]["f1"],
                "validation_joint_accuracy": validation["slots"]["joint_intent_slot_accuracy"],
                "validation_false_accept_rate": validation["scope"][
                    "unsupported_false_accept_rate"
                ],
                "validation_false_reject_rate": validation["scope"]["in_scope_false_reject_rate"],
                "real_test_intent_macro_f1": (
                    real_test["intent"]["macro"]["f1"] if real_test else None
                ),
                "real_test_joint_accuracy": (
                    real_test["slots"]["joint_intent_slot_accuracy"] if real_test else None
                ),
                "real_test_false_accept_rate": (
                    real_test["scope"]["unsupported_false_accept_rate"] if real_test else None
                ),
                "real_test_false_reject_rate": (
                    real_test["scope"]["in_scope_false_reject_rate"] if real_test else None
                ),
                "onnx_size_bytes": (result.get("onnx") or {}).get("size_bytes"),
                "onnx_cpu_latency_p95_ms": (result.get("onnx") or {})
                .get("cpu_latency_two_second_audio_ms", {})
                .get("p95"),
            }
        )
    atomic_json(output_root / f"{stem}.json", rows)
    with (output_root / f"{stem}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def fixed_condition_summary(
    results: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Aggregate pre-registered seeds without using test metrics for selection."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result["data_condition"]), []).append(result)
    rows = []
    for condition, values in sorted(grouped.items()):
        validation = [row["validation"] for row in values]
        scope = [row["scope"] for row in validation]
        rows.append(
            {
                "architecture": str(values[0]["architecture"]),
                "data_condition": condition,
                "rejection_strategy": str(values[0]["rejection_strategy"]),
                "seeds": sorted(int(row["seed"]) for row in values),
                "validation_intent_macro_f1_mean": float(
                    np.mean([row["intent"]["macro"]["f1"] for row in validation])
                ),
                "validation_joint_accuracy_mean": float(
                    np.mean([row["slots"]["joint_intent_slot_accuracy"] for row in validation])
                ),
                "validation_false_accept_rate_mean": float(
                    np.mean([row["unsupported_false_accept_rate"] for row in scope])
                ),
                "validation_false_reject_rate_mean": float(
                    np.mean([row["in_scope_false_reject_rate"] for row in scope])
                ),
                "validation_constraints_met_for_every_seed": all(
                    row["unsupported_false_accept_rate"]
                    <= float(config["selection"]["maximum_unsupported_false_accept_rate"])
                    and row["in_scope_false_reject_rate"]
                    <= float(config["selection"]["maximum_in_scope_false_reject_rate"])
                    for row in scope
                ),
            }
        )
    return rows


def fixed_condition_order(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["validation_constraints_met_for_every_seed"]),
        float(row["validation_intent_macro_f1_mean"]),
        float(row["validation_joint_accuracy_mean"]),
        -float(row["validation_false_reject_rate_mean"]),
        -float(row["validation_false_accept_rate_mean"]),
    )


def run_fixed_condition_ablation(
    project_root: Path,
    config: dict[str, Any],
    device: torch.device,
    architectures: list[str],
    conditions: list[str],
    smoke_test: bool,
) -> dict[str, Any]:
    """Run a small controlled data ablation with fixed, prior-selected settings.

    This mode deliberately has no HPO. It compares only predeclared data
    conditions, chooses from validation aggregates, and records locked-test
    results afterwards without using them for deployment selection.
    """
    if architectures != ["tiny_conformer"]:
        raise ValueError("fixed_condition_ablation is restricted to tiny_conformer")
    if not conditions:
        raise ValueError("fixed_condition_ablation needs at least one data condition")
    if set(config["rejection_strategies"]) != {"binary_scope"}:
        raise ValueError("fixed_condition_ablation is restricted to binary_scope")

    output_root = resolve(project_root, config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_root = resolve(project_root, config["paths"]["prepared_dataset"])
    release = validate_release(prepared_root)
    initialize_study_identity(output_root, config, prepared_root / "release.json")
    atomic_json(output_root / "dataset_release.json", release)
    run_onnx_preflight(output_root, config, architectures)

    seeds = (
        [int(config["seed"])]
        if smoke_test
        else [int(seed) for seed in config["replication_seeds"]]
    )
    started = time.time()
    results = []
    matrix = [(condition, seed) for condition in conditions for seed in seeds]
    for index, (condition, seed) in enumerate(matrix, start=1):
        identity = {
            "architecture": "tiny_conformer",
            "data_condition": condition,
            "rejection_strategy": "binary_scope",
            "seed": seed,
        }
        atomic_json(
            output_root / "status.json",
            {
                "state": "training_fixed_ablation",
                "experiment": index,
                "experiments": len(matrix),
                "current": identity,
                "elapsed_seconds": time.time() - started,
            },
        )
        try:
            results.append(
                run_training_job(
                    project_root,
                    config,
                    device,
                    **identity,
                    output=run_directory(output_root, identity),
                    smoke_test=smoke_test,
                )
            )
        except Exception as error:  # noqa: BLE001
            directory = run_directory(output_root, identity)
            atomic_json(
                directory / "status.json",
                {
                    **identity,
                    "state": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
    if len(results) != len(matrix):
        raise RuntimeError("A fixed ablation run failed; inspect its status.json before selection")

    validation_summary = fixed_condition_summary(results, config)
    atomic_json(output_root / "validation_summary.json", validation_summary)
    write_comparison(output_root, results, "validation_comparison")
    winner = max(validation_summary, key=fixed_condition_order)
    selected_condition = str(winner["data_condition"])

    # The two conditions and seeds were pre-registered. Test metrics are
    # reported for the complete ablation but never affect the chosen condition.
    atomic_json(output_root / "status.json", {"state": "locked_test_evaluation"})
    tested = [
        evaluate_test(
            project_root,
            config,
            device,
            result,
            run_directory(output_root, result),
            smoke_test,
        )
        for result in results
    ]
    write_comparison(output_root, tested, "test_comparison")
    deployment_candidates = [
        row for row in tested if row["data_condition"] == selected_condition
    ]
    deployment_run = max(deployment_candidates, key=lambda row: validation_order(row, config))
    selection = {
        "selection_source": "two-seed validation-only fixed-condition ablation",
        "test_used_for_selection": False,
        "winner": winner,
        "deployment_seed": deployment_run["seed"],
        "model": str(run_directory(output_root, deployment_run) / "model.onnx"),
        "metadata": str(run_directory(output_root, deployment_run) / "deployment_metadata.json"),
        "holdout_evaluated": False,
        "holdout_next_step": (
            "Run scripts/evaluate_holdout.py on the Raspberry Pi 5 after accepting this ablation."
        ),
    }
    atomic_json(output_root / "selection.json", selection)
    atomic_json(
        output_root / "status.json",
        {
            "state": "complete",
            "elapsed_seconds": time.time() - started,
            "selection": str(output_root / "selection.json"),
        },
    )
    return selection


def run_study(
    project_root: Path,
    config: dict[str, Any],
    device: torch.device,
    architectures: list[str],
    conditions: list[str],
    smoke_test: bool,
) -> dict[str, Any]:
    if config.get("study_mode") == "fixed_condition_ablation":
        return run_fixed_condition_ablation(
            project_root, config, device, architectures, conditions, smoke_test
        )
    return run_full_study(project_root, config, device, architectures, conditions, smoke_test)


def run_full_study(
    project_root: Path,
    config: dict[str, Any],
    device: torch.device,
    architectures: list[str],
    conditions: list[str],
    smoke_test: bool,
) -> dict[str, Any]:
    output_root = resolve(project_root, config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    prepared_root = resolve(project_root, config["paths"]["prepared_dataset"])
    release = validate_release(prepared_root)
    initialize_study_identity(output_root, config, prepared_root / "release.json")
    atomic_json(output_root / "dataset_release.json", release)
    run_onnx_preflight(output_root, config, architectures)
    started = time.time()
    atomic_json(output_root / "status.json", {"state": "optimizing", "started": started})
    tuned = {
        architecture: tune_architecture(
            project_root, config, device, architecture, output_root, smoke_test
        )
        for architecture in architectures
    }

    comparison_results = []
    strategy = str(config["comparison"]["reference_rejection_strategy"])
    matrix = [
        (architecture, condition)
        for architecture in architectures
        for condition in conditions
    ]
    for index, (architecture, condition) in enumerate(matrix, start=1):
        identity = {
            "architecture": architecture,
            "data_condition": condition,
            "rejection_strategy": strategy,
            "seed": int(config["seed"]),
        }
        atomic_json(
            output_root / "status.json",
            {
                "state": "comparing",
                "experiment": index,
                "experiments": len(matrix),
                "current": identity,
                "elapsed_seconds": time.time() - started,
            },
        )
        try:
            comparison_results.append(
                run_training_job(
                    project_root,
                    config,
                    device,
                    **identity,
                    output=run_directory(output_root, identity),
                    training_overrides=tuned[architecture],
                    smoke_test=smoke_test,
                )
            )
        except Exception as error:  # noqa: BLE001
            directory = run_directory(output_root, identity)
            atomic_json(
                directory / "status.json",
                {
                    **identity,
                    "state": "failed",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )

    if not comparison_results:
        raise RuntimeError("Every architecture/data comparison failed")
    finalists = sorted(
        comparison_results,
        key=lambda result: validation_order(result, config),
        reverse=True,
    )[: int(config["comparison"]["rejection_finalists"])]
    rejection_results = list(comparison_results)
    for finalist in finalists:
        for rejection in config["rejection_strategies"]:
            identity = {
                **{key: finalist[key] for key in ("architecture", "data_condition", "seed")},
                "rejection_strategy": rejection,
            }
            if any(run_identity(row) == run_identity(identity) for row in rejection_results):
                continue
            try:
                rejection_results.append(run_training_job(
                    project_root,
                    config,
                    device,
                    **identity,
                    output=run_directory(output_root, identity),
                    training_overrides=tuned[identity["architecture"]],
                    smoke_test=smoke_test,
                ))
            except Exception as error:  # noqa: BLE001
                directory = run_directory(output_root, identity)
                atomic_json(
                    directory / "status.json",
                    {
                        **identity,
                        "state": "failed",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                )

    unique = {run_identity(row): row for row in rejection_results}
    write_comparison(output_root, list(unique.values()))
    replication_bases = sorted(
        unique.values(), key=lambda result: validation_order(result, config), reverse=True
    )[: int(config["comparison"]["replication_finalists"])]
    replicated = []
    seeds = (
        [int(config["seed"])]
        if smoke_test
        else [int(value) for value in config["replication_seeds"]]
    )
    for base in replication_bases:
        for seed in seeds:
            identity = {
                "architecture": base["architecture"],
                "data_condition": base["data_condition"],
                "rejection_strategy": base["rejection_strategy"],
                "seed": seed,
            }
            try:
                replicated.append(run_training_job(
                    project_root,
                    config,
                    device,
                    **identity,
                    output=run_directory(output_root, identity),
                    training_overrides=tuned[identity["architecture"]],
                    smoke_test=smoke_test,
                ))
            except Exception as error:  # noqa: BLE001
                directory = run_directory(output_root, identity)
                atomic_json(
                    directory / "status.json",
                    {
                        **identity,
                        "state": "failed",
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    },
                )

    expected_seeds = len(seeds)
    complete_identities = {
        identity
        for identity in {run_identity(row) for row in replicated}
        if sum(run_identity(row) == identity for row in replicated) == expected_seeds
    }
    replicated = [row for row in replicated if run_identity(row) in complete_identities]
    if not replicated:
        raise RuntimeError("No finalist completed every pre-registered replication seed")

    atomic_json(output_root / "status.json", {"state": "locked_test_evaluation"})
    tested = [
        evaluate_test(
            project_root,
            config,
            device,
            result,
            run_directory(output_root, result),
            smoke_test=smoke_test,
        )
        for result in replicated
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for result in tested:
        grouped.setdefault(run_identity(result), []).append(result)
    aggregate = []
    for identity, rows in grouped.items():
        scores = [row["test"]["real"]["intent"]["macro"]["f1"] for row in rows]
        joint = [row["test"]["real"]["slots"]["joint_intent_slot_accuracy"] for row in rows]
        false_accept = [
            row["test"]["all"]["scope"]["unsupported_false_accept_rate"] for row in rows
        ]
        false_reject = [row["test"]["real"]["scope"]["in_scope_false_reject_rate"] for row in rows]
        aggregate.append(
            {
                "architecture": identity[0],
                "data_condition": identity[1],
                "rejection_strategy": identity[2],
                "seeds": [row["seed"] for row in rows],
                "real_test_macro_f1_mean": float(np.mean(scores)),
                "real_test_macro_f1_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "real_test_joint_accuracy_mean": float(np.mean(joint)),
                "test_false_accept_rate_mean": float(np.mean(false_accept)),
                "real_test_false_reject_rate_mean": float(np.mean(false_reject)),
            }
        )
    def final_order(row: dict[str, Any]) -> tuple[float, ...]:
        feasible = (
            row["test_false_accept_rate_mean"]
            <= float(config["selection"]["maximum_unsupported_false_accept_rate"])
            and row["real_test_false_reject_rate_mean"]
            <= float(config["selection"]["maximum_in_scope_false_reject_rate"])
        )
        return (
            float(feasible),
            row["real_test_macro_f1_mean"],
            row["real_test_joint_accuracy_mean"],
            -row["test_false_accept_rate_mean"],
            -row["real_test_false_reject_rate_mean"],
        )

    winner = max(aggregate, key=final_order)
    winner_runs = [row for row in tested if run_identity(row) == run_identity(winner)]
    deployment_run = max(winner_runs, key=lambda row: validation_order(row, config))
    selection = {
        "winner": winner,
        "deployment_seed": deployment_run["seed"],
        "model": str(run_directory(output_root, deployment_run) / "model.onnx"),
        "metadata": str(run_directory(output_root, deployment_run) / "deployment_metadata.json"),
        "holdout_evaluated": False,
        "holdout_next_step": "Run scripts/evaluate_holdout.py on the Raspberry Pi 5.",
    }
    atomic_json(output_root / "selection.json", selection)
    atomic_json(output_root / "replicated_test_summary.json", aggregate)
    write_comparison(output_root, tested, "finalist_test_comparison")
    atomic_json(
        output_root / "status.json",
        {
            "state": "complete",
            "elapsed_seconds": time.time() - started,
            "selection": str(output_root / "selection.json"),
        },
    )
    return selection

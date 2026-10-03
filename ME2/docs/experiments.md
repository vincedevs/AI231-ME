# VCM experiment workflow

The repository retains the compact evidence for the reported completed studies
under `results/experiments/`. The larger output directories shown below are
generated workspaces and are excluded from Git. Re-running a pinned
configuration recreates them.

`scripts/run_experiments.py` implements the full, resumable model-selection
study. It runs sequentially because only one GPU may be visible.

## Stages

1. Verify the prepared release and all manifest hashes.
   A study-identity guard rejects an output directory created from another
   dataset release or experiment configuration, so stale HPO trials and
   checkpoints cannot be silently reused.
2. Preflight ONNX export for DS-CNN, TC-ResNet, BC-ResNet, CRNN-attention, and
   Tiny Conformer before spending GPU time.
3. Tune each architecture with 15 Optuna TPE trials and median pruning on the
   same `supplemented_real + binary_scope` reference condition.
4. With each architecture's fixed tuned hyperparameters, compare all five
   architectures across five data conditions using the binary scope head.
5. For the two validation finalists, compare confidence, internal UNKNOWN, and
   binary-scope rejection.
6. Replicate the two best validation configurations using seeds 231, 947, and
   2027.
7. Evaluate only those replicated finalists on the locked test set, separately
   reporting real, synthetic, and combined strata.
8. Select and export the winning raw-waveform ONNX artifact. Do not evaluate
   the holdout here.

HPO search ranges, seeds, constraints, augmentation, optimizer, epochs, and
audio features are explicit in `configs/experiments.json`. Training uses
AdamW, cosine decay, gradient clipping, deterministic waveform augmentation,
mixed precision on CUDA, class-aware replacement sampling, and early stopping.

## Commands

```bash
# Plan only; no dataset or model is loaded
python scripts/run_experiments.py --dry-run

# Recommended preflight after dataset preparation
python scripts/run_experiments.py --device cuda --smoke-test \
  --architectures dscnn --data-conditions supplemented_real

# Smoke-test artifacts are written beside the full study at
# models/experiments/ai231_me2_voice_commands_smoke/.

# Complete resumable one-GPU study
python -u scripts/run_experiments.py --device cuda 2>&1 | tee experiments.log

# Status from another shell
python scripts/show_experiment_status.py --watch
```

Under tmux, run the foreground command directly. Detach with `Ctrl-b d` and
inspect with `tail -f experiments.log`; `nohup` is unnecessary.

## Outputs

```text
models/experiments/ai231_me2_voice_commands/
  optimization/<architecture>/{study.db,trials.csv,best_hyperparameters.json}
  runs/<architecture>/<condition>/<rejection>/seed_<seed>/
  comparison.{json,csv}
  finalist_test_comparison.{json,csv}
  replicated_test_summary.json
  selection.json
  status.json
```

Each run records resolved configuration, environment, dataset-release hash,
dataset counts, checkpoints, epoch history, validation/test predictions,
calibration and thresholds, metrics, plots, ONNX parity, model size, parameter
count, and CPU latency. Existing completed runs are reused and incomplete
training resumes from `last.pt`.

A failed independent run is recorded with its traceback. Final selection
requires at least one finalist to complete every registered replication seed,
so partial evidence cannot silently become the reported winner.

## Final holdout

Only after `selection.json` is final should the Pi run:

```bash
python scripts/evaluate_holdout.py \
  --selection models/experiments/ai231_me2_voice_commands/selection.json \
  --output models/final_holdout \
  --confirm-final
```

The output directory must not already exist. This discourages repeated
holdout-guided iteration and records release, manifest, selection, and model
hashes with predictions, metrics, latency, and real-time factor.

## Time-budgeted study

When less than 12 hours remain for both DGX training and Raspberry Pi testing,
use `configs/experiments_budgeted.json`. This predeclared study compares
TC-ResNet and Tiny Conformer under three matched conditions:

1. screened real speech only (`supplemented_real`);
2. the same development pool plus benchmark synthetic speech
   (`supplemented_mixed`); and
3. that mixed pool plus the supplemental shard's training-voice partition
   (`supplemented_mixed_extended`).

It uses six short Optuna trials per architecture, fixes low-priority search
dimensions, retains binary-scope rejection from the earlier rejection study,
trains each controlled comparison with seed 231, and replicates only the
validation winner with seed 947. This is a budgeted confirmatory comparison,
not evidence that the other three architectures are inferior on the revised
dataset.

```bash
python scripts/run_experiments.py \
  --config configs/experiments_budgeted.json \
  --dry-run

python scripts/run_experiments.py \
  --config configs/experiments_budgeted.json \
  --device cuda \
  --smoke-test

python -u scripts/run_experiments.py \
  --config configs/experiments_budgeted.json \
  --device cuda 2>&1 | tee experiments-budgeted.log

python scripts/show_experiment_status.py \
  --output models/experiments/ai231_me2_voice_commands_budgeted \
  --watch
```

The completed one-A100 study took 2.44 hours. Runtime on another GPU or storage
system depends on early stopping and I/O throughput. Reserve additional time
for copying the selected ONNX model and running the sealed holdout once on the
Raspberry Pi 5.

## Tiny Conformer synthetic-negative ablation

`configs/experiments_tiny_conformer_negatives.json` is a narrow rejection
ablation, not a replacement model-selection study. It uses the prior-selected
Tiny Conformer hyperparameters without HPO and compares two pre-registered
conditions across seeds 231 and 947:

1. `supplemented_mixed`, the selected baseline; and
2. `supplemented_mixed_with_negatives`, the same data plus 1,000 pinned,
   training-only `synthetic_negatives/train` source clips labeled `UNSUPPORTED`.

One exact decoded-PCM duplicate in that source is deterministically excluded,
leaving 999 unique negative training clips. The exclusion is recorded in the
prepared release report rather than weakening the project's duplicate-audio
leakage guard.

The negative split is loaded from the Hugging Face revision recorded in
`configs/dataset_negative_ablation.json`. Its separate 250-clip test split is
not materialized, trained on, or used for calibration. The official test and
the Raspberry Pi holdout remain unchanged. The runner chooses the condition
from the two-seed validation aggregate only; it writes official-test metrics
for both pre-registered conditions afterwards as report-only evidence.

```bash
uv sync --frozen --python 3.12 --extra dataset --extra training --extra dev

python scripts/prepare_dataset.py \
  --config configs/dataset_negative_ablation.json

python scripts/prepare_dataset.py \
  --config configs/dataset_negative_ablation.json \
  --validate-only

CUDA_VISIBLE_DEVICES=0 python -u scripts/run_experiments.py \
  --config configs/experiments_tiny_conformer_negatives.json \
  --device cuda \
  --gpu-index 0 2>&1 | tee tiny-conformer-negative-ablation.log

# From a second shell
python scripts/show_experiment_status.py \
  --output models/experiments/tiny_conformer_negative_ablation \
  --watch
```

It runs four training jobs (two conditions × two seeds), sequentially on one
visible GPU. The outputs are isolated in
`models/experiments/tiny_conformer_negative_ablation/`, including
`validation_summary.json`, `validation_comparison.{json,csv}`,
`test_comparison.{json,csv}`, and `selection.json`.

Curated copies of the completed budgeted and negative-ablation evidence are in
`results/experiments/budgeted/` and `results/experiments/negative_ablation/`.
They include the selection records, metrics, predictions, configurations,
calibration evidence, and plots needed to audit the reported conclusions.

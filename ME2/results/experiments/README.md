# Curated experiment evidence

This directory contains the compact evidence needed to audit the quantitative
claims in the root README without committing large, reproducible training
workspaces or PyTorch checkpoints.

## Contents

- `budgeted/` contains the final TC-ResNet versus Tiny Conformer study,
  hyperparameter-search records, selected configurations, validation/test
  predictions and metrics, plots, ONNX preflight checks, and deployment
  metadata.
- `negative_ablation/` contains the Tiny Conformer synthetic-negative ablation,
  its two-seed comparisons, calibration records, predictions, plots, and
  deployment metadata.

The original complete outputs were generated under `models/experiments/`.
That path is intentionally ignored because it contains large checkpoints and
intermediate products. Re-run the pinned configuration files from the root
README to regenerate the full workspace.

These files are results, not alternative configuration sources. The canonical
inputs remain `configs/`, the prepared dataset release identity, and the source
code used by the experiment scripts.

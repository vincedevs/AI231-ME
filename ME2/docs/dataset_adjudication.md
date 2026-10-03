# Dataset label adjudication

## Purpose

The frozen Hugging Face test membership is never changed. Label review and the
single duplicate-audio exclusion are recorded in sidecar files so the source
Parquet files and retained previous-release manifests remain untouched.

`configs/dataset_label_overrides.jsonl` contains corrections to the frozen
Hugging Face labels. `configs/supplement_label_overrides.jsonl` contains
reviewed transformations of previous-release utterance groups.
`configs/dataset_exclusions.jsonl` records the later of two identical Common
Voice holdout rows in source revision
`da92a79ffde3031d5bb2a25138d9dd7d9f7ed006`; excluding it prevents duplicate
weighting while preserving the raw release for audit.

## Reminder review policy

The supplemental review is restricted to SLURP `calendar_set` utterances that
were previously treated as unsupported. An utterance is mapped to
`CREATE_REMINDER` only when it explicitly requests a reminder, notification,
warning, or equivalent “do not forget” behavior.

Ordinary calendar creation, calendar queries, calendar synchronization, event
repetition, timers, incomplete commands, and conditional or multi-event
requests remain unsupported. Absence from the override sidecar means that the
original unsupported label is retained.

Each accepted row records:

- source dataset and original intent;
- immutable utterance-group identity;
- verbatim transcript;
- canonical intent;
- exact transcript surface used for the canonical `task` slot; and
- adjudication rationale.

The decision is applied at utterance-group level, so close/distant microphone
recordings receive the same label and cannot be split across development
partitions. The override transforms the existing record; it never appends a
second copy with a conflicting label.

## Leakage and duplicate controls

Before a reviewed supplement is admitted, preparation excludes reliable
speakers present in the frozen test or holdout and rejects duplicate source
identities or identical decoded PCM audio. Assignment then keeps every
reliable speaker and recording group in one split. Synthetic speech and data
without trustworthy speaker identity remain training-only.

The prepared release copies all three sidecars and records their SHA-256 hashes in
`release.json`. `reports/supplement_selection.json` records every reviewed
record as selected, excluded for evaluation-speaker overlap, rejected for a
specific integrity reason, or not needed after the target was met.

For the current reviewed sidecar, 36 utterance groups matched 61 retained
recordings. Evaluation-speaker screening excluded 7 groups (13 recordings).
The release selected 29 groups (48 recordings): 37 recordings entered training
and 11 entered validation. Together with the frozen benchmark data, validation
contains 15 `CREATE_REMINDER` recordings from 5 speakers and 10 independent
recording groups.

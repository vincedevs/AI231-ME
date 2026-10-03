# VCM contract

## Canonical output

- The canonical schema is Option A from `primer/Dataset Schema.xlsx`.
- The model has exactly 19 action intents and six slot types.
- `PLAY_MUSIC`, `CALL`, and `MESSAGE` are slotless in schema version 1.0.0.
- Song/artist, contact, and message-content slots are deferred.
- English is the only language in the primary benchmark.
- `UNKNOWN` and `NO_COMMAND` are not canonical intent classes.
- Alfred's action integrations are independent of this training pipeline; the
  VCM output contract does not embed credentials or execute side effects.
- Raw audio is immutable. The prepared release contains checksummed normalized
  copies so it remains portable across the development machine, DGX, and Pi.

## Unsupported commands

A 19-way classifier is closed-set: without a rejection mechanism, it must map
every input to one of the 19 intents. Therefore, intent softmax confidence alone
must not be treated as proof that a command is supported.

The application contract has three decisions: `execute`, `low_confidence`, and
`unsupported`. Before deployment, the VCM pipeline must calibrate this decision
on held-out unsupported speech, including semantically close commands such as
“turn on the fan.” Useful negative speech already exists among the out-of-scope
FSC, SLURP, and STOP records. Environmental audio can supplement evaluation but
cannot substitute for difficult spoken negatives.

The experiment plan compares confidence-only rejection, an internal twentieth
`UNKNOWN` output, and a separate binary in-scope head. `UNKNOWN` is an
experimental model output only; it does not change the 19-intent application
schema. The binary in-scope head is the proposed primary design.

Only `decision=execute` may reach an action handler. Missing required slots,
low confidence, or an unsupported-input score below its validation-selected
threshold must not execute an action.

## Deferred schema scope

- Richer slots for music, contacts, and message content.
- Multilingual experiments.

Synthetic speech already present in the pinned benchmark is included only in
the explicit mixed-data conditions. Validation remains real-only, and test
metrics are stratified into real and synthetic results.

Any schema or frozen-benchmark change requires a version increment.

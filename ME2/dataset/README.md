# Reproducible VCM dataset

The authoritative benchmark input is the pinned Hugging Face release in
`dataset/raw/ai231_me2_voice_commands/`. The preparation configuration records
the revision, eleven Parquet hashes, expected row counts, schema, adjudications,
split seed, and supplementation targets. Everything below `dataset/` except
this README is local-only and excluded from Git.

For a clean, single-command reproduction of the public benchmark, run:

```bash
uv run python scripts/reproduce_dataset.py --benchmark-only
```

This writes `dataset/prepared_benchmark_only/`; it does not overwrite a full
prepared experiment release.

The reported training study also used the private supplemental provenance
release
[`vincedevs/alfred-vcm-supplement`](https://huggingface.co/datasets/vincedevs/alfred-vcm-supplement)
at revision `c376ada37c0eea13a4db1893cc854a347e5d25b5`. It has exact metadata for
all 16,450 selected supplemental records, but redistributes audio for only the
5,355 SLURP and ESC-50 records whose terms permit this derived release. The
remaining 11,095 STOP, Fluent Speech Commands, SNIPS, and Rochester records are
metadata-only. Obtain those original datasets under their own terms and hydrate
each manifest's missing `audio_path` before installation. The expected local
download is:

```text
dataset/raw/alfred_vcm_supplement/
  export_summary.json
  source_inventory.csv
  manifests/{train,validation}.jsonl
  audio/...                         # 5,355 redistributed clips initially
```

Hydration is deterministic but cannot be automated without accepting each
upstream dataset's terms. For every row where `audio_included` is false:

1. identify the original recording using `source_dataset` and `source_id`;
2. decode it, resample to mono 16 kHz PCM16 using the same normalization in
   `src/dataset_pipeline/prepare.py`, and confirm that the resulting decoded PCM
   hash equals the row's `pcm_sha256`;
3. write that WAV at the row's `prepared_audio_path`, relative to
   `dataset/raw/alfred_vcm_supplement/`; and
4. in the **local downloaded manifest only**, set `audio_path` to that same
   relative `prepared_audio_path` value.

Do not commit the hydrated audio or locally edited manifests. If a source item
cannot reproduce its recorded PCM hash, stop: substituting another utterance
would no longer reproduce the reported experiment.

Once every one of the 16,450 manifest rows resolves to an audio file inside
that release, reproduce the complete reported dataset with:

```bash
uv run python scripts/download_dataset.py --with-supplement-release
# Acquire and hydrate the restricted-source rows before the next command.
uv run python scripts/install_supplement_release.py
uv run python scripts/prepare_dataset.py --overwrite
uv run python scripts/prepare_dataset.py --validate-only
```

The installer deliberately fails on the first missing audio item instead of
silently producing a smaller or methodologically different dataset. If the
historical fully hydrated source already exists at `dataset/training/`, the
same workflow is available as one command:

```bash
uv run python scripts/reproduce_dataset.py --skip-download
```

The individual preparation commands are:

```bash
uv run python scripts/prepare_dataset.py
uv run python scripts/prepare_dataset.py --validate-only
```

Use `--without-supplements` only for the frozen-benchmark-only ablation.
Use `--overwrite` to intentionally rebuild an existing prepared release.
Content-addressed normalized audio from an interrupted or previous build is
checksum-verified and reused during an overwrite; unreferenced audio is pruned
after a successful rebuild.

## Output

`dataset/prepared/` is portable between the development machine, DGX, and
Raspberry Pi:

```text
audio/<sha-prefix>/<pcm-sha256>.wav
manifests/{train,validation,test,holdout}.jsonl
reports/
schema.json
label_overrides.jsonl
dataset_exclusions.jsonl
supplement_label_overrides.jsonl
release.json
```

Every manifest and audio item has a checksum. Audio is normalized to mono
16 kHz PCM16 and stored once by decoded PCM hash. Records preserve source,
speaker, transcript, source ID, mapping rule, synthetic/supplemental flags, and
canonical labels.

## Scientific controls

- Official test membership is unchanged. The prepared holdout contains 201 of
  the 202 source rows because one exact duplicate is removed through
  `configs/dataset_exclusions.jsonl`; the source Parquet remains untouched.
- The holdout is marked sealed and the training loader refuses to open it.
- Reviewed corrections live in `configs/dataset_label_overrides.jsonl`; the
  original Parquet files are untouched.
- Reviewed relabeling of previous-release samples lives in
  `configs/supplement_label_overrides.jsonl`. Its source provenance,
  transcript, canonical intent, task annotation, and rationale are pinned and
  copied into every prepared release.
- Validation is real-only and speaker-disjoint wherever speaker identity is
  trustworthy. STOP pseudo-speakers are explicitly treated as unreliable and
  kept out of validation.
- Every validation label requires at least five real examples except
  `MESSAGE`, for which the release contains only seven real, speaker-identified
  development recordings across four speakers. Its explicit floor is two
  records, two speakers, and two recording groups per development partition;
  the selected validation split contains three recordings from two speakers.
  `CREATE_REMINDER` additionally requires at least five speakers and ten
  independent recording groups in both development partitions.
- Previous-release supplements are real, development-only, and screened
  against evaluation speakers, source identities, and exact decoded-audio
  hashes. Reliable speakers may enter train or validation; unreliable speaker
  identities remain training-only.
- Synthetic benchmark audio is available only in explicit mixed-data
  conditions. It never enters validation.
- The `supplemental_synth` Parquet shard is hash-verified. Only its 3,983
  `voice_split=train` rows are materialized, marked as supplemental synthetic,
  and exposed through the separate `supplemented_mixed_extended` condition.
  Its test-voice and holdout-voice partitions are excluded. The existing four
  conditions retain their prior membership, so the extra TTS is a controlled
  training-only ablation rather than a silent dataset change.
- The auxiliary numerals split is excluded because it has no command-level
  intent label. It can support a future slot-only study, but must not silently
  enter the present intent experiment.
- Dataset preparation records split counts, source revision, overrides,
  rejected supplements, overlaps, slot-annotation coverage, and all manifest
  hashes. Missing slots remain explicitly unannotated; they are never replaced
  with invented targets.

The old `dataset/training/` directory is retained only as the reproducible
source of approved real supplements. It is not the new experiment release and
must not be uploaded to the DGX in place of `dataset/prepared/`.

## Synthetic-negative ablation release

`configs/dataset_negative_ablation.json` creates the separate
`dataset/prepared_negative_ablation/` release. It preserves every item in the
main prepared release and adds 999 unique `OUT_OF_SCOPE` clips from the pinned
1,000-row Hugging Face `synthetic_negatives/train` split. One exact decoded
PCM duplicate is excluded and recorded in the release report. They are synthetic,
supplemental, and training-only. Existing data conditions exclude them; only
`supplemented_mixed_with_negatives` includes them.

The Hugging Face `synthetic_negatives/test` split is deliberately excluded from
the release: it is an external rejection diagnostic, never a training,
validation, calibration, official-test, or holdout source.

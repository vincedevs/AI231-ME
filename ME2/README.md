# ME2 — ALFRED: Voice Controlled Smart Device

Alfred is an English, edge-oriented voice assistant built for a master's-level
machine-learning project. Its primary research component is a Voice Command
Model (VCM) trained from scratch to classify supported commands, extract the
project's six canonical slots, and reject unsupported speech. The complete
application adds wake-word detection, voice-activity detection, follow-up
speech recognition, action execution, and spoken/audio feedback for deployment
on a Raspberry Pi 5.

This README is the entry point for reproducing the dataset, experiments,
reported results, and final application. Component-specific detail is linked
where it is useful rather than duplicated here.

> **Licence:** original project source code is released under the
> [MIT License](LICENSE). This licence does not relicense datasets, pretrained
> models, generated model weights, voices, or other third-party assets. Those
> materials remain subject to their documented upstream terms; Alfred's
> third-party notices are retained under `alfred/licenses/`.

## System overview

```text
microphone audio
  -> OpenWakeWord-compatible "Hey Alfred" / "I'm Batman" detector
  -> Silero VAD command capture
  -> Tiny Conformer VCM
       -> 19-intent classification
       -> character-CTC slot extraction
       -> binary in-scope score
  -> confidence and scope policy
  -> action execution
  -> Piper TTS and/or audio feedback
  -> return to wake-word listening
```

Moonshine Tiny is an optional, application-level ASR component used only for
clarification prompts such as reminder text, message recipient/content, or a
weather location. It is not the VCM and does not replace the trained intent and
slot outputs.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/dataset_pipeline/` | Dataset download, normalization, validation, leakage checks, and reproducible release creation |
| `src/vcm_training/` | Features, models, training, calibration, evaluation, and ONNX export |
| `scripts/` | Auditable entry points for dataset preparation and experiments |
| `configs/` | Pinned dataset identities and explicit experiment configurations |
| `results/experiments/` | Compact, committed evidence for the reported DGX experiments |
| `results/raspberry_pi/` | Full physical Raspberry Pi benchmark report and raw trial records |
| `alfred/` | Deployable Raspberry Pi 5 assistant and its runtime assets |
| `wayne_manor/` | Separate local smart-device API and dashboard used by Alfred actions |
| `dataset_reviewer/` | Optional read-only TUI for the legacy packaged supplement source layout |
| `dataset/README.md` | Exact dataset acquisition, expected layout, and reproduction limits |
| `docs/` | Research protocol, contracts, adjudication records, deployment, and model provenance |
| `tests/` | Dataset/training pipeline tests |
| `me2-deck.pdf` | Final project submission deck |

Raw datasets, prepared audio, complete experiment workspaces, checkpoints,
credentials, local databases, logs, and environment directories are deliberately
not committed. See [Data and reproducibility](#data-and-reproducibility).

## Canonical VCM contract

The authoritative schemas are `primer/Dataset Schema.xlsx` and
`primer/Intent Schema.xlsx`. The deployed VCM supports 19 intents:

```text
PLAY_MUSIC         WEATHER            TIME               LIGHT_ON
LIGHT_OFF          PAUSE              STOP               NEXT
VOLUME_UP          VOLUME_DOWN        CALL               MESSAGE
LIST_REMINDERS     TIMER              ALARM              TEMPERATURE
BRIGHTNESS         COLOR              CREATE_REMINDER
```

The six canonical slot types are `duration`, `time`, `degrees`, `percent`,
`color`, and `task`. `PLAY_MUSIC`, `CALL`, and `MESSAGE` are intentionally
slotless in VCM version 1.0; their richer information is collected by an
application-level follow-up when required.

Unsupported speech is a rejection outcome, not a twentieth executable intent.
The model predicts the 19 canonical intents in the exact order stored in its
deployment metadata; it does **not** predict 93 separate variation classes.
The physical benchmark's 93 supported phrase/slot combinations are scoring
cases mapped onto those 19 intents. Application code must load the metadata
alongside the ONNX graph and must not infer label order from directories.

For the exact tensor names, dimensions, post-processing, threshold semantics,
and command-log format, see [the VCM contract](docs/vcm_contract.md).

## Requirements and dependency setup

The repository targets **Python 3.12** and uses
[uv](https://docs.astral.sh/uv/) lock files. Do not share a virtual environment
between the research pipeline, Alfred, and Wayne Manor: each component has an
independent dependency lock.

Install uv and confirm it is on `PATH`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

### Research and training environment

From the repository root:

```bash
uv sync --frozen --python 3.12 --extra dataset --extra training --extra dev
source .venv/bin/activate
```

`uv.lock` is authoritative. `requirements.txt` is a compatibility entry point
for environments where uv cannot be used. On a managed accelerator host,
install the site's supported CUDA/PyTorch build if it differs from the locked
wheel and record the exact environment with the experiment output.

### Alfred development environment

```bash
cd alfred
uv sync --frozen --python 3.12 --extra dev
cp .env.example .env
uv run python -m alfred --self-check
```

The Raspberry Pi procedure and operating-system packages are documented in
[`alfred/deploy/RASPBERRY_PI_SETUP.md`](alfred/deploy/RASPBERRY_PI_SETUP.md).
Use `alfred/deploy/initialize_pi.sh` only after completing those manual system
installation steps.

### Wayne Manor development environment

Wayne Manor is intentionally run independently from Alfred:

```bash
cd wayne_manor
uv sync --frozen --python 3.12 --extra dev
cd frontend
npm ci
npm run build
cd ..
uv run python -m wayne_manor
```

## Data and reproducibility

### Pinned source datasets

The primary benchmark is the Hugging Face dataset
[`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands),
pinned to revision:

```text
da92a79ffde3031d5bb2a25138d9dd7d9f7ed006
```

Its source membership is 10,733 train, 4,443 test, 202 sealed holdout, and
66,390 numeral records. One exact decoded-audio duplicate is excluded from the
prepared holdout by a reviewed, checksummed sidecar, leaving 201 prepared
holdout clips. The official test and holdout memberships are otherwise
preserved.

The exact training supplements used by the reported experiment are published
as [`vincedevs/alfred-vcm-supplement`](https://huggingface.co/datasets/vincedevs/alfred-vcm-supplement),
pinned to revision:

```text
c376ada37c0eea13a4db1893cc854a347e5d25b5
```

That release records 16,450 supplement rows: 15,037 assigned to training and
1,413 assigned to validation. It includes 5,355 redistributable audio files
from SLURP and ESC-50. The remaining 11,095 STOP, Fluent Speech Commands,
SNIPS, and Rochester rows are metadata-only because their source audio cannot
be redistributed here. Full bit-for-bit reproduction therefore requires legal
access to those original datasets and hydration of the recorded source paths.
This limitation is explicit rather than silently replacing unavailable audio.

### Download and prepare

All data stays under the ignored `dataset/` directory. The expected paths and
source-specific assumptions are described in [`dataset/README.md`](dataset/README.md).

To reproduce the public benchmark without private/restricted supplements:

```bash
uv run python scripts/reproduce_dataset.py --benchmark-only
```

To reproduce the complete reported training release:

```bash
uv run python scripts/download_dataset.py --with-supplement-release
# Hydrate metadata-only supplement rows from their original licensed datasets.
uv run python scripts/install_supplement_release.py
uv run python scripts/prepare_dataset.py --overwrite
uv run python scripts/prepare_dataset.py --validate-only
```

If the historical source tree already exists under `dataset/training/`, the
equivalent convenience command is:

```bash
uv run python scripts/reproduce_dataset.py --skip-download
```

Preparation is deterministic from the pinned revisions and sidecars. It:

1. validates source schemas and checksums;
2. reads labels from Hugging Face metadata, never folder names;
3. normalizes audio to mono 16 kHz PCM without modifying originals;
4. applies reviewed label adjudications;
5. removes exact decoded-audio duplicates and checks cross-split leakage;
6. preserves official test and holdout membership;
7. builds a real-only, speaker-disjoint validation split;
8. keeps derived/augmented examples in the parent sample's logical split; and
9. writes checksummed JSONL manifests and a release audit under
   `dataset/prepared/`.

The reported prepared release contained 35,810 clips (23.73 hours):

| Split | Clips | Hours | Speakers | Notes |
| --- | ---: | ---: | ---: | --- |
| Train | 29,406 | 19.722 | 423 | Real and explicitly tagged supplemental/synthetic data |
| Validation | 1,760 | 1.372 | 48 | Real-only and speaker-disjoint from training |
| Test | 4,443 | 2.454 | 144 | Frozen source membership; 824 real and 3,619 synthetic |
| Holdout | 201 | 0.178 | 7 | Sealed until final device evaluation |

The selected `supplemented_mixed` training view contains 25,423 records:
23,468 supported, 1,955 unsupported, 8,374 synthetic, and 9,434 with slot
annotations. Numeral data and arbitrary supplemental datasets are not mixed in
automatically; inclusion is controlled by the experiment configuration.

## VCM architecture and preprocessing

The selected model is a compact Tiny Conformer operating on raw mono waveform.
The exported ONNX graph includes its feature extraction, so training and Alfred
share the same preprocessing:

- sample rate: 16,000 Hz;
- log-mel bins: 40;
- FFT/window: 400 samples (25 ms);
- hop: 160 samples (10 ms);
- mel band: 20–7,600 Hz;
- two stride-2 1-D convolutional subsampling layers;
- four Conformer blocks, model dimension 96, four attention heads;
- full-context pooling for classification;
- 19-way intent classifier, character-CTC slot head, and binary scope head;
- 936,349 trainable parameters;
- ONNX opset 17, approximately 4.55 MB.

The joint training objective is:

```text
intent cross-entropy + 0.5 * slot CTC loss + 0.5 * scope binary cross-entropy
```

Waveform augmentation is training-only: probability 0.8, gain up to ±6 dB,
time shift up to 100 ms, and additive noise at 15–35 dB SNR. No augmented or
supplement-generated derivative is allowed to cross its logical split.

## Experiments and model selection

### Budgeted architecture/data comparison

The final study in `configs/experiments_budgeted.json` compared TC-ResNet and
Tiny Conformer under identical preprocessing and three data conditions:

- `supplemented_real`;
- `supplemented_mixed`; and
- `supplemented_mixed_extended`.

Each architecture received six Optuna TPE trials of up to 12 epochs on the
validation split. Final candidates used seed 231, and the winning condition was
replicated with seed 947. The study used one NVIDIA A100 GPU (`cuda:0`) and
completed in 2.44 hours. The sealed holdout was not used for tuning or model
selection.

Validation results:

| Architecture | Data condition | Intent macro F1 | Joint accuracy | False rejection rate |
| --- | --- | ---: | ---: | ---: |
| TC-ResNet | supplemented_real | 0.639 | 0.500 | 0.704 |
| TC-ResNet | supplemented_mixed | 0.735 | 0.564 | 0.624 |
| TC-ResNet | supplemented_mixed_extended | 0.718 | 0.551 | 0.603 |
| Tiny Conformer | supplemented_real | 0.671 | 0.566 | 0.594 |
| **Tiny Conformer** | **supplemented_mixed** | **0.739** | **0.631** | **0.471** |
| Tiny Conformer | supplemented_mixed_extended | 0.724 | 0.618 | 0.532 |

The selected Tiny Conformer used:

| Parameter | Value |
| --- | ---: |
| Learning rate | 0.0009741372095369595 |
| Weight decay | 0.000002300985610287488 |
| Batch size | 64 |
| Label smoothing | 0.05 |
| Gradient clipping | 5.0 |
| Maximum epochs / patience | 35 / 6 |
| Unsupported sampling fraction | 0.15 |
| Mixed precision | Enabled |

On the frozen test set, seed 231 achieved intent macro F1 0.739, joint
intent-slot accuracy 0.653, false-accept rate 0.108, and false-reject rate
0.449. Seed 947 achieved 0.656, 0.568, 0.000, and 0.637 respectively. The
two-seed means were 0.698 macro F1 and 0.611 joint accuracy, showing material
seed sensitivity that should not be hidden by reporting only the best run.

The combined frozen test (which is synthetic-heavy) produced 0.949 intent
accuracy and 0.927 macro F1 for seed 231. This number is reported separately
because it is not directly comparable to the real-speech rejection metrics.

### Synthetic-negative ablation

A controlled ablation added 999 unique synthetic out-of-scope samples to
training only and repeated seeds 231 and 947. On validation, the mean intent
macro F1 changed from 0.728 to 0.722, joint accuracy from 0.620 to 0.620, and
false-rejection rate from 0.549 to 0.521. The modest rejection improvement did
not justify the macro-F1 regression, so the baseline condition remained the
selected deployment model.

### Rejection calibration

Thresholds were selected on validation data only, targeting at least 0.95
action precision and at most 0.05 unsupported-command execution. The deployed
seed-231 metadata records:

```text
intent confidence threshold: 0.866007924079895
binary in-scope threshold:    0.9999032967447421
```

These are calibrated policy values, not arbitrary defaults. Test and holdout
labels are never used to choose them.

Compact tables, configurations, calibration output, predictions, plots, and
deployment metadata supporting these claims are committed under
[`results/experiments/`](results/experiments/). Complete checkpoints and
training workspaces are reproducible but intentionally excluded from Git.

## Reproducing training and evaluation

After preparing and validating the dataset:

```bash
uv run python scripts/run_experiments.py \
  --config configs/experiments_budgeted.json \
  --dry-run

CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiments.py \
  --config configs/experiments_budgeted.json \
  --device cuda

uv run python scripts/show_experiment_status.py --watch
```

The runner restricts itself to one visible GPU. Its output records the dataset
identity, random seed, hyperparameters, history, validation/test predictions,
metrics, plots, threshold calibration, ONNX export, and deployment metadata.
The experiment order is load, validate, preprocess, split, define, train,
predict, evaluate, analyze, and save; important methodological choices remain
visible in the scripts and JSON configurations.

Run the additional negative-data experiment independently:

```bash
uv run python scripts/prepare_dataset.py \
  --config configs/dataset_negative_ablation.json

CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_experiments.py \
  --config configs/experiments_tiny_conformer_negatives.json \
  --device cuda
```

Do not evaluate the sealed holdout during development. After selection is
irrevocably complete, use the documented one-time command in
[`docs/benchmark_protocol.md`](docs/benchmark_protocol.md).

## Included runtime model files

Only models required by supported Alfred behavior are committed. Every other
ONNX file is rejected by `.gitignore` unless explicitly allowlisted.

| Model | Purpose |
| --- | --- |
| `alfred/assets/models/vcm/21c7b228fd63-20559251ca67/model.onnx` | Current selected Tiny Conformer VCM |
| `alfred/assets/models/vcm/3a2a4c8540bd-67de9defe703/model.onnx` | Documented runtime comparison/rollback selectable by `--vcm-variant 3a2a` |
| `alfred/assets/models/wake_word/hey_alfred.onnx` | Custom primary wake phrase head |
| `alfred/assets/models/wake_word/im_batman.onnx` | Optional easter-egg wake phrase head |
| `alfred/assets/models/wake_word/melspectrogram.onnx` | Shared OpenWakeWord mel frontend |
| `alfred/assets/models/wake_word/embedding.onnx` | Shared OpenWakeWord speech embedding frontend |
| `alfred/assets/models/vad/silero_vad.onnx` | Silero voice-activity detector |
| `alfred/assets/models/tts/vits-piper-en_GB-alan-medium/en_GB-alan-medium.onnx` | Piper British-English feedback voice |

Associated JSON/configuration and licence files are retained beside those
graphs. Moonshine Tiny is optional and downloaded from its pinned upstream
release rather than committed:

```bash
cd alfred
uv run python tools/install_clarification_asr.py
```

## Running Alfred

Copy `alfred/.env.example` to `alfred/.env` and set only the services you use:

| Variable | Meaning |
| --- | --- |
| `ALFRED_WAYNE_MANOR_URL` | Base URL for the separately running Wayne Manor device API |
| `ALFRED_LINPHONE_DESTINATION` | SIP destination for `CALL` |
| `ALFRED_GOOGLE_CHAT_WEBHOOK_URL` | Private Google Chat webhook for `MESSAGE` |

Never commit `.env`, SIP accounts, webhooks, tokens, or keys.

Development/mock run:

```bash
cd alfred
uv run python -m alfred --mock
```

Raspberry Pi service lifecycle after installation and initialization:

```bash
cd ~/alfred
alfred run
alfred status
alfred stop
```

Useful non-destructive checks:

```bash
uv run python -m alfred --self-check
uv run python -m alfred --list-devices
uv run python -m alfred --once
```

See [`alfred/README.md`](alfred/README.md) for device selection, service setup,
actions, feedback, benchmark mode, and troubleshooting.

## Raspberry Pi 5 physical benchmark

The final full benchmark used the external
[`airimonda/vcm-benchmark`](https://github.com/airimonda/vcm-benchmark) harness
at commit:

```text
ab39857cf73dd067d068fed388c321ab160dc191
```

It played all 202 Hugging Face holdout commands from a laptop speaker into the
ReSpeaker microphone connected to the Raspberry Pi 5, plus 16 no-wake negative
trials. Alfred ran in benchmark mode, which logs predictions and timing without
executing physical actions or speaking feedback. The run lasted 60.1 minutes.

Key measured results:

| Metric | Result |
| --- | ---: |
| Wake detections | 90.1% |
| False wake detections | 0 / 16 |
| Intent accuracy | 52.5% |
| Intent macro precision / recall / F1 | 93.7% / 53.3% / 62.6% |
| End-to-end command accuracy | 49.5% |
| Out-of-scope false-accept rate | 6.2% (1 / 16) |
| False-reject rate | 50.5% |
| Slot exact match | 87.8% (49 slot-bearing trials) |
| Real / synthetic intent accuracy | 40.6% / 63.2% |
| VCM inference mean / p95 | 8.8 ms / 12.4 ms |
| Real-time factor p95 | 0.006 |
| End-to-end response p95 | 1.679 s |
| Peak resident memory | 543.3 MB |
| Maximum CPU temperature | 56.8 °C, no throttling |

The benchmark's generic ONNX graph counter reports 1,104,613 stored tensor
elements, while the training pipeline reports 936,349 trainable parameters.
These answer different questions and should not be presented as contradictory.
No standalone long-duration wake-word false-accepts-per-hour evaluation is
retained; the 0/16 result is specific to the physical command benchmark.

Reproduce the harness environment on the playback/control machine:

```bash
git clone https://github.com/airimonda/vcm-benchmark.git
git -C vcm-benchmark checkout ab39857cf73dd067d068fed388c321ab160dc191
python3.12 -m venv vcm-benchmark/.venv
vcm-benchmark/.venv/bin/pip install -r vcm-benchmark/requirements.txt
```

Start Alfred on the Pi:

```bash
cd ~/alfred
alfred run --benchmark
```

Then run the full physical benchmark from the harness checkout:

```bash
.venv/bin/python benchmark.py \
  --mode ssh \
  --host raspy@raspy.local \
  --log-dir '~/vcm_benchmark' \
  --size full
```

The committed report, configuration, per-trial data, and Pi telemetry are in
[`results/raspberry_pi/vcm_benchmark_20261003/`](results/raspberry_pi/vcm_benchmark_20261003/).

## Verification

Research pipeline:

```bash
uv run ruff check src scripts tests
uv run python -m pytest -q
```

Alfred:

```bash
cd alfred
uv run ruff check src tests tools
uv run python -m pytest -q
uv run python -m alfred --self-check
```

Wayne Manor:

```bash
cd wayne_manor
uv run ruff check src tests
uv run python -m pytest -q
cd frontend
npm ci
npm test
npm run build
```

## Important decisions and limitations

- English audio is the benchmark scope.
- Speaker generalization is primary; phrase generalization is secondary.
- Validation is real-only and speaker-disjoint where speaker identity exists.
- Synthetic speech is training-only for controlled conditions, except where
  the externally supplied frozen test membership itself contains synthetic
  audio; results distinguish real and synthetic subsets.
- The holdout is sealed until final Raspberry Pi assessment and is never used
  for HPO, calibration, or model selection.
- The selected model shows substantial seed and real-speech sensitivity.
- Slot annotations are sparse and inconsistent across source corpora; the
  application therefore requests clarification rather than silently executing
  uncertain arguments.
- Physical acoustic performance depends on speaker placement, room acoustics,
  playback level, microphone gain, and competing Spotify output.
- A long-duration wake-word false-accepts-per-hour study remains future work.
- Full supplement reproduction requires original licensed audio for the
  metadata-only rows described above.
- External services (Spotify Soloist, SIP/baresip, Google Chat, Open-Meteo, and
  Wayne Manor) require independent setup and may fail without network access or
  credentials. Alfred exposes explicit failure feedback rather than hiding it.

## Further documentation

- [Dataset preparation and provenance](dataset/README.md)
- [Experiment methodology and commands](docs/experiments.md)
- [Physical benchmark protocol](docs/benchmark_protocol.md)
- [Dataset adjudication record](docs/dataset_adjudication.md)
- [VCM runtime contract](docs/vcm_contract.md)
- [OpenWakeWord model provenance](docs/openwakeword_models.md)
- [Alfred application guide](alfred/README.md)
- [Raspberry Pi installation](alfred/deploy/RASPBERRY_PI_SETUP.md)
- [Wayne Manor guide](wayne_manor/README.md)

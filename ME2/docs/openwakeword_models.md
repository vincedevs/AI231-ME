# Custom OpenWakeWord models

## Scope and provenance

Alfred has two project-specific wake phrases:

- `Hey Alfred` -> `alfred/assets/models/wake_word/hey_alfred.onnx`
- `I'm Batman` -> `alfred/assets/models/wake_word/im_batman.onnx`

These are custom OpenWakeWord classifier models, not stock OpenWakeWord wake
words. They were produced using OpenWakeWord's custom-model workflow and were
then copied into this repository for self-contained Raspberry Pi deployment.
The OpenWakeWord source is:

- https://github.com/dscripka/openWakeWord
- automated training notebook:
  https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb
- example custom-model configuration:
  https://github.com/dscripka/openWakeWord/blob/main/examples/custom_model.yml

The prior implementation retains an audited OpenWakeWord checkout at commit
`368c03716d1e92591906a84949bc477f3a834455`. The original phrase-specific YAML
files, generated-data manifests, training histories, and held-out predictions
were not retained. That commit therefore documents the audited implementation,
but it must not be presented as a cryptographically proven training revision.

## What was trained

OpenWakeWord separates inference into three components:

```text
16 kHz mono PCM
  -> shared OpenWakeWord mel-spectrogram model
  -> shared frozen OpenWakeWord speech embedding model
  -> custom binary classifier for one wake phrase
  -> probability in [0, 1]
```

Only the phrase classifiers are project-specific. The frontend and embedding
network are pretrained OpenWakeWord resources and remain shared between the two
classifiers. Consequently, it is accurate to say that we **trained custom wake
word models using OpenWakeWord transfer learning**. It is not accurate to say
that the entire wake-word stack was trained from scratch.

Each packaged classifier accepts a `[1, 16, 96]` embedding window and emits one
probability with shape `[1, 1]`. They are two separate ONNX models rather than a
single model with two output heads. Keeping them separate lets the application
share frontend computation while enabling independent thresholds and model
replacement.

## OpenWakeWord training workflow used

The OpenWakeWord automated workflow used for each phrase is:

1. Define one target phrase and a unique model name in a copy of
   `examples/custom_model.yml`.
2. Generate thousands of positive English examples with the workflow's Piper
   sample generator. Keep generated training and validation samples separate.
3. Define confusable phrases as adversarial negatives and add large,
   phrase-absent feature datasets containing speech, noise, and music.
4. Apply independent acoustic augmentation to the synthetic speech, including
   background mixing and room impulse responses.
5. Convert positive and negative audio through the frozen OpenWakeWord
   mel-spectrogram and embedding models.
6. Train the small binary classifier with OpenWakeWord's automated training
   routine, which increases negative weighting while monitoring validation
   performance and false activations.
7. Export the phrase classifier to ONNX and test it with the exact shared
   frontend used at deployment.

The upstream process can be reproduced in a separate checkout as follows. The
configuration values below are intentionally placeholders: the original values
for these two artifacts were not preserved and must not be reconstructed by
guessing.

```bash
git clone https://github.com/dscripka/openWakeWord.git
cd openWakeWord

# Follow the environment and dataset setup in the upstream automated notebook.
# Create one YAML for each target from examples/custom_model.yml:
#   hey_alfred.yml: model_name=hey_alfred, target_phrase=["hey alfred"]
#   im_batman.yml:  model_name=im_batman,  target_phrase=["i'm batman"]
# Also configure positive counts, validation counts, background/RIR paths,
# negative feature files, training steps, and false-positive target.

# Run each stage explicitly so its inputs and failures remain auditable.
python openwakeword/train.py --training_config path/to/hey_alfred.yml --generate_clips
python openwakeword/train.py --training_config path/to/hey_alfred.yml --augment_clips
python openwakeword/train.py --training_config path/to/hey_alfred.yml --train_model

python openwakeword/train.py --training_config path/to/im_batman.yml --generate_clips
python openwakeword/train.py --training_config path/to/im_batman.yml --augment_clips
python openwakeword/train.py --training_config path/to/im_batman.yml --train_model
```

Before using these commands for a future retraining, verify their arguments
against the pinned upstream revision. OpenWakeWord's training interface is an
external dependency and can change.

## Packaged artifacts

| Artifact | Role | SHA-256 |
| --- | --- | --- |
| `melspectrogram.onnx` | shared OpenWakeWord audio frontend | `ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f` |
| `embedding.onnx` | shared frozen speech embedding | `70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f` |
| `hey_alfred.onnx` | custom `Hey Alfred` binary classifier | `5402e10b10e946243949064463254428b4c7c05fa6ac9a9c96b5c022f1064aa6` |
| `im_batman.onnx` | custom `I'm Batman` binary classifier | `8096621acc33b95c9a013aa203b11eb27ee29782fe0e009c675bf3c17d45a195` |

The authoritative deployment manifest is
`alfred/assets/models/manifest.json`. Alfred's streaming implementation is in
`alfred/src/alfred/wake_word.py`.

## Runtime policy is not training evidence

The application currently evaluates 1,280 samples (80 ms at 16 kHz) per
streaming step, applies a threshold of `0.50` to each classifier, requires two
consecutive positive frames, and uses a 1.5-second cooldown. Silero VAD is a
runtime false-activation guard; it is not part of either custom classifier and
was not used to create their labels.

These values are application operating parameters. They do not establish model
accuracy and should be tuned only on validation recordings representative of
the Raspberry Pi 5 and ReSpeaker environment.

## Evidence limitations and required evaluation

Because the original training configuration, split manifests, histories, and
held-out predictions are absent, this repository cannot reproduce the exact
training run or honestly report its training/validation accuracy. It also
cannot derive false accepts per hour from the ONNX files alone.

The existing ONNX artifacts should be frozen and evaluated with:

- speaker-diverse, real positive recordings for both phrases;
- close/far, quiet/noisy, and varied microphone-angle conditions;
- phonetically similar hard negatives;
- many hours of deployment-like speech, music, and environmental sound;
- validation-only selection of threshold, consecutive hits, and VAD settings;
- one untouched test set for final false-reject rate, false accepts per hour,
  detection latency, and uncertainty intervals.

This is an evaluation task, not a new scratch-training experiment. Until those
data exist, the defensible claims are limited to artifact integrity, interface
compatibility, runtime integration, and measured inference latency.

## Licensing

OpenWakeWord code is Apache-2.0 licensed. Its repository states that included
pretrained models are CC BY-NC-SA 4.0 because of their training-data licenses.
The repository retains the applicable OpenWakeWord notice and license text in
`alfred/THIRD_PARTY_NOTICES.md` and `alfred/licenses/`.

# Dataset redistribution policy

## Purpose

`scripts/prepare_dataset.py` builds the exact local prepared release used by
the VCM study. It combines the pinned Hugging Face benchmark release with
screened, nonduplicate supplemental recordings from earlier locally obtained
source datasets.

This document distinguishes **local research use** from **public redistribution
of audio**. A source's permission to train a research model does not imply
permission to upload its raw or transformed audio to a new Hugging Face
repository.

## Non-overlap guarantee

The preparation pipeline starts with the source identity and decoded 16 kHz
PCM hash of every pinned benchmark record. A supplemental candidate is rejected
when it repeats either one. It also excludes reliable speakers in the frozen
test or holdout and audits the final release for split leakage.

The checked prepared release has:

- 35,810 records and 35,810 unique decoded PCM clips;
- no duplicate source identities;
- no reliable-speaker overlap across train, validation, test, and holdout; and
- no recording-group overlap across those splits.

The supplement search rejected 960 duplicate-audio candidates. Source corpora
such as SLURP, Fluent Speech Commands, and SNIPS occur in both the benchmark
and supplement pools, but no admitted recording duplicates the benchmark
audio.

## Supplemental audio in the selected training condition

The selected `supplemented_mixed` condition contains 15,037 non-synthetic
supplemental clips:

| Source | Clips | Public raw-audio release in a derived HF dataset? | Reason |
| --- | ---: | --- | --- |
| STOP | 6,151 | No | Its license forbids modification, incorporation into another dataset, and distribution of its audio except for up to ten clips in a research or academic publication. |
| SLURP | 4,649 | Conditional | The audio license is CC BY-NC 4.0. Publication requires attribution, non-commercial terms, and a review that the chosen Hugging Face access and license communicate those obligations. |
| Fluent Speech Commands | 2,927 | No for Alfred's transformed subset | FSC is CC BY-NC-ND 4.0 and restricted to academic research. It permits noncommercial sharing of the unmodified material with attribution, but Alfred resamples audio and changes labels, which creates an adapted derivative. |
| SNIPS SLU | 1,154 | No | Re-publication is permitted only when datasets are unmodified and remain under the same terms. Alfred normalizes audio and changes labels. |
| Rochester Smart Speaker Commands | 146 | Not until its authoritative license is retained and reviewed | The local source provenance does not include a redistribution license. |
| ESC-50 | 10 | Conditional | ESC-50 is CC BY-NC 3.0 as a whole and requires the relevant attribution notice for each source clip. |

The table concerns the **additional supplements**. It does not claim that the
already-published benchmark release has one uniform license: every upstream
source retains its own terms.

## What can be published now

Publish a separate Hugging Face **provenance and rebuild** repository, not a
public mirror of the merged audio. It may contain:

- `configs/dataset.json` and the frozen schema;
- label overrides, exclusions, and supplemental-label overrides;
- source IDs, original source labels, canonical labels, slots, split labels,
  PCM hashes, and recording-group IDs;
- source acquisition instructions and required citations/licenses;
- preparation code and a checksummed release manifest; and
- aggregate counts and evaluation reports.

It must not contain STOP audio. It must not contain SNIPS or FSC audio after
Alfred's normalization/relabeling. Rochester audio remains excluded until its
redistribution terms are documented. A private Hugging Face repository is not
an exception: uploading there still distributes a copy to Hugging Face.

## Reproducing the full local study release

Obtain every external source under its own terms, place it in the paths
declared by `configs/dataset.json`, then run:

```bash
python scripts/prepare_dataset.py --config configs/dataset.json --overwrite
python scripts/prepare_dataset.py --config configs/dataset.json --validate-only
```

The first command is the single dataset-generation entry point. The second
verifies the prepared release without changing it. No raw source audio is
copied into the repository by either command.

## Primary license evidence

- STOP Dataset License Agreement, section 6.
  https://dl.fbaipublicfiles.com/stop/LICENSE.txt
- SLURP `LICENSE.txt`.
  https://github.com/pswietojanski/slurp/blob/master/LICENSE.txt
- SNIPS Spoken Language Understanding research datasets license summary.
  https://github.com/sonos/spoken-language-understanding-research-datasets
- ESC-50 `LICENSE`.
  https://github.com/karolpiczak/ESC-50/blob/master/LICENSE
- Fluent Speech Commands Public License.
  https://fluent.ai/wp-content/uploads/2021/04/Fluent_Speech_Commands_Public_License.pdf

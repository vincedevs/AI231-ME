# DJ dataset reviewer

DJ is a read-only terminal application for browsing the legacy packaged
supplement source, inspecting annotations, and playing referenced audio. It is
kept in this standalone folder so it does not become part of dataset generation
or model training. It is an inspection aid, not an input to the current
preparation pipeline and not evidence that local data are committed.

The application requires a locally reproduced `dataset/training` tree (ignored
by Git), defaults to that path, uses the
`speaker_generalization` protocol, and includes the unsupported-command and
Kokoro training manifests. It never modifies the selected dataset.

## Install

From the repository root, using Python 3.12:

```bash
python3.12 -m venv .reviewer-venv
source .reviewer-venv/bin/activate
python -m pip install -r dataset_reviewer/requirements.txt
```

## Run

```bash
python dataset_reviewer/dj.py
```

Review the phrase-generalization protocol instead:

```bash
python dataset_reviewer/dj.py --protocol phrase_generalization
```

Exclude auxiliary manifests when a narrower view is useful:

```bash
python dataset_reviewer/dj.py --no-synthetic --no-unsupported
```

Use another packaged dataset root:

```bash
python dataset_reviewer/dj.py --dataset /path/to/dataset/training
```

Run `python dataset_reviewer/dj.py --help` for all options.

## Controls

| Control | Action |
| --- | --- |
| Arrow keys or mouse | Select a record |
| Enter or Space | Play selected audio |
| `s` | Stop playback |
| `[` / `]` | Previous or next page |
| `/` | Focus search |
| `r` | Select a random matching record |
| `c` | Clear all filters |
| `q` | Stop playback and quit |

DJ automatically uses `afplay` on macOS. It can also use `ffplay`, `mpv`,
`paplay`, or `aplay` when installed.

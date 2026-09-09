# 🎵 Music Analyzer v2

[![Python](https://img.shields.io/badge/python-3.13-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.13-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]()

A **multi-label music genre & mood (feeling) classifier**, trained on the
[MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) dataset.

Given a raw audio file, the project extracts a rich set of acoustic features and
predicts, at the same time, **which genres** the song belongs to (e.g. `rock`,
`electronic`, `jazz`) **and** **which mood/feeling** it conveys (e.g. `happy`,
`sad`, `energetic`).

---

## ✨ Features

| Module | What it does |
|---|---|
| `audio_loader` | librosa-based loading with validation & clear errors |
| `feature_extractor` | log-Mel spectrograms, MFCC / chroma / spectral-contrast sequences, and 38 scalar statistics (RMS, tempo, key/mode, dynamic range, ...) |
| `create_dataset` / `create_labels` | turn raw MTG-Jamendo files into compressed `.npz` feature files + `labels.json` |
| `models` | per-segment **CNN encoder + Bi-GRU**, fused with a **scalar-features MLP**; separate multi-label heads for genres and feelings |
| `train` | AdamW + ReduceLROnPlateau, early stopping, checkpoints (saves the fitted scalar normalizer too) |
| `demo` | interactive CLI that classifies any audio file with a saved model |
| `supervised_create_dataset.cpp` | low-level watchdog that keeps dataset creation running through crashes |

**Model inputs**
- `log_mel_segments` — `(batch, segments, 1, n_mels=128, frames)`
- `scalar_features` — `(batch, 38)` (mean/min/max/std/... over 5 low-level arrays
  + duration, harmonic/percussive ratio, dynamic range, key, mode, tempo)

**Model outputs** — multi-hot predictions over:
- **15 genres** (rock, pop, electronic, hip-hop, techno, jazz, metal, rap, ...)
- **14 feelings** (happy, sad, energetic, calm, dark, epic, melancholic, ...)

---

## 🚀 Installation

Requires **Python ≥ 3.13** and uses [`uv`](https://docs.astral.sh/uv/).

```bash
# install dependencies
uv sync

# + dev tools (ruff linter)
uv sync --group dev
```

| Task | Command |
|---|---|
| Train the model | `jupyter notebook src/music_analyzer_v2/handle_training.ipynb` |
| Run the interactive demo | `uv run python -m music_analyzer_v2.demo` |

---

## 📦 Dataset pipeline

The [MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) dataset provides
~55k full audio tracks tagged with genres, instruments and mood/theme. This
project focuses on a curated subset and maps the raw MTG tags onto a compact,
consistent label set.

### 1. Build the labels
`create_labels.py` reads the MTG **split TSV** files, keeps only tracks present
in both the genre and mood subsets, filters out tracks whose `.npz` is **missing
or corrupted**, maps raw tags to the project's target labels, and writes
`parsed_dataset/labels.json`:

```bash
cd src/music_analyzer_v2
python create_labels.py            # dry-run (builds labels.json)
python create_labels.py --delete    # also delete corrupt .npz files
```

### 2. Extract features to `.npz`
Each track becomes one compressed `.npz` file (named after the track id) inside
`parsed_dataset/`. Extraction is **multi-process** and **resumable** (already
parsed tracks are skipped):

```bash
cd src/music_analyzer_v2
python create_dataset.py
```

Because extraction is CPU-heavy and can occasionally crash on a specific file,
the optional C++ **supervisor** automatically re-runs it, tracks progress by
counting `.npz` files, and stops cleanly when nothing is left to do:

```bash
cd scripts
g++ -std=c++17 -O2 -o supervised_create_dataset.exe supervised_create_dataset.cpp
./supervised_create_dataset.exe            # run until dataset is complete
./supervised_create_dataset.exe --del-parsed   # ... and clean mp3s afterwards
```

When you want to free disk space after parsing, delete the already-parsed mp3s:

```bash
python delete_parsed_mp3.py            # dry-run
python delete_parsed_mp3.py --confirm  # actually delete
```

> ℹ️ The raw `mtg-jamendo-dataset/` and the generated `parsed_dataset/` are
> git-ignored — only code is shared, data stays local.
---

## 🧠 Model architecture

```
Audio ──extract──► log-Mel segments ──► [CNN encoder] ─► Bi-GRU ─►┐
                  scalar features (38) ─► [MLP] ────────────────►├─► genre head
                                                                 └─► feeling head
```

- Each log-Mel **segment** is embedded by a compact CNN
  (`Conv2D → BatchNorm → GELU → MaxPool`, ×4, ending in an adaptive average pool).
- The segment embeddings are processed by a **bidirectional GRU**.
- The 38 scalar features are processed by a small **MLP** and fused with the
  sequence representation.
- Two **multi-label heads** (genre & feeling) produce logits, trained with
  `BCEWithLogitsLoss`.

**Training recipe** (`train.py`): AdamW, `lr = 1e-3`, weight decay `1e-3`,
gradient clipping at `5.0`, early stopping after 7 epochs without validation
improvement, and checkpointing (model + optimizer + scalar normalizer), so
training can be resumed.

---

## 🗂️ Project layout

```
music-analyzer_v2/
├── pyproject.toml                # project metadata, deps, entry point
├── scripts/
|   ├── supervised_create_dataset.cpp # C++ watchdog for dataset creation
|   └── delete_parsed_mp3.py          # free disk space by removing parsed mp3s
└── src/music_analyzer_v2/
    ├── audio_loader.py           # AudioData + load_audio
    ├── feature_extractor.py      # ExtractedFeatures pipeline
    ├── create_dataset.py         # .npz feature extraction
    ├── create_labels.py          # labels.json generation
    ├── dataset.py                # MusicDataset / MusicDataLoader / collate
    ├── models.py                 # MusicAnalyzerModel0 + load_saved_model
    ├── train.py                  # run_epoch / train / checkpoints
    ├── test_model.py             # evaluation after training
    ├── demo.py                   # interactive inference CLI
    ├── handle_training.ipynb     # end-to-end training notebook
    └── utils.py                  # label lists, thresholds, helpers
```

---

## 🧪 Evaluation

`test_model.py` runs the usual `run_epoch` on the **test** split (no optimizer)
and reports the same metrics used during training: overall loss, genre loss,
feeling loss, and **micro-F1** for genres and feelings.

---

## 📄 License

This project is released under the **MIT License**.

The underlying **MTG-Jamendo Dataset** is *not* included in this repository.
It is made available for **non-commercial research** use only (audio is licensed
under Creative Commons, metadata under CC BY-NC-SA); see the
[dataset page](https://mtg.github.io/mtg-jamendo-dataset/) for full terms.

---

## 🙏 Credits

Built with [PyTorch](https://pytorch.org/) and [librosa](https://librosa.org/),
trained on the [MTG-Jamendo](https://mtg.github.io/mtg-jamendo-dataset/) dataset
by the Music Technology Group (UPF) — Automatic Music Tagging.

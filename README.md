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

> ☁️ Για ολόκληρη τη ροή στο **Google Colab** (download + τοπικά `.npz` + upload στο
> Drive) δες το [`colab/dataset_pipeline.ipynb`](colab/dataset_pipeline.ipynb).
>
> Όλα τα paths προσδιορίζονται από τη θέση του package και μπορούν να
> παρακαμφθούν με env vars: `MA2_RAW_DATASET`, `MA2_PARSED_DATASET`,
> `MA2_LABELS_PATH`, `MA2_CHECKPOINT_DIR`, `MA2_SAVED_MODELS_DIR`.

### 1. Build the labels
`create_labels.py` reads the MTG **split TSV** files, keeps only tracks present
in both the genre and mood subsets, filters out tracks whose `.npz` is **missing
or corrupted**, maps raw tags to the project's target labels, and writes
`parsed_dataset/labels.json`:

```bash
python -m music_analyzer_v2.create_labels                       # build labels.json
python -m music_analyzer_v2.create_labels --check --delete      # + keep only valid .npz
```

The split TSVs are **metadata** and are not served by the audio CDN — they live
in the official GitHub repository. Fetch them with:

```bash
python scripts/download_mtg_jamendo.py --mode splits
# -> <raw dataset>/data/splits/split-0/autotagging_{genre,moodtheme}-{train,validation,test}.tsv
```

**Split-0 coverage** (verified against the official TSVs):

| subset | train | validation | test | total |
|---|---|---|---|---|
| `autotagging_genre` | 32 572 | 11 043 | 11 479 | **55 094** |
| `autotagging_moodtheme` | 9 949 | 3 802 | 4 231 | **17 982** |
| in **both** (used here) | 9 810 | 3 767 | 4 159 | **17 736** |

Tracks that exist in only one subset (37 358 genre-only, 246 mood-only) are
**not** part of this project, because the model predicts genres *and* feelings
simultaneously and therefore needs both label sets.

Of the 17 736 tracks present in both subsets, the ones whose tags do not map
onto the project's target labels are dropped as well (a track must end up with
at least one genre *and* one feeling). This leaves the final manifest:

| split | tracks |
|---|---|
| train | 4 742 |
| validation | 1 799 |
| test | 1 916 |
| **total** | **8 457** |

### 2. Download the raw audio
`scripts/download_mtg_jamendo.py` fetches **only the tracks listed in
`labels.json`** from the official CDN — parallel, resumable (Range) and
verified against `Content-Length`:

```bash
python scripts/download_mtg_jamendo.py --dry-run    # show what would be fetched
python scripts/download_mtg_jamendo.py --limit 10   # quick test
python scripts/download_mtg_jamendo.py              # subset, low quality
python scripts/download_mtg_jamendo.py --mode tars  # whole dataset, low (~165 GB)
python scripts/download_mtg_jamendo.py --quality full --mode tars  # full 320 kbps (~540 GB)
```

> ⚠️ **Always use the universal `raw_30s` source** — the downloader does this
> automatically:
>
> ```
> https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/<NN>/<id>.low.mp3
> ```
>
> The per-subset paths (`autotagging_genre/audio-low/...`,
> `autotagging_moodtheme/audio-low/...`) must **not** be used: they return
> **HTTP 404** for tracks that are not part of that specific subset, even
> though the very same file exists under `raw_30s`. Since `raw_30s` holds the
> whole corpus, it covers every track of both subsets, and tracks belonging to
> both are downloaded only once.

The split TSVs reference the full-quality files (`41/241.mp3`); the low-quality
CDN serves the same audio as `41/241.low.mp3`. The conversion is done on the
*string* level (never `Path.with_suffix()`, which would turn `241.low.mp3` into
`241.low.low.mp3`), so it is idempotent and safe for names with several dots.

> The project loads audio as **mono @ 22.05 kHz** (`audio_loader.py`), so the
> mono `.low.mp3` variant (default) keeps essentially all usable content while
> being ~3.4× smaller — use `--quality full` only if you need 320 kbps/stereo
> sources (then also adapt the `.low.mp3` suffix in `create_labels.py`).

**Disk requirements** (for the 8 457-track manifest):

| item | size |
|---|---|
| raw audio, low quality (`.low.mp3`) | ~33 GB (measured ≈ 3.9 MB/track) |
| raw audio, full quality (320 kbps stereo) | ~100 GB |
| parsed `.npz` features | ~65 GB (measured ≈ 7.7 MB/track) |

> 💡 The `.npz` files are **larger** than the audio they are derived from. Parse
> and delete the mp3s in batches (`python scripts/delete_parsed_mp3.py
> --confirm`) if disk space is tight.

### 3. Extract features to `.npz`
Each track becomes one compressed `.npz` file (named after the track id) inside
`parsed_dataset/`. Extraction is **multi-process** and **resumable** (already
parsed tracks are skipped):

```bash
python -m music_analyzer_v2.create_dataset
```

Because extraction is CPU-heavy and can occasionally crash on a specific file,
the **portable Python supervisor** re-runs it, tracks progress by counting
`.npz` files in the real output directory, stops cleanly when nothing is left
to do, and can upload every new `.npz` to a target directory (e.g. a Google
Drive mount) after each round:

```bash
python scripts/supervised_create_dataset.py                      # run until done (local)
python scripts/supervised_create_dataset.py --upload-to <dir>    # + incremental upload
python scripts/supervised_create_dataset.py --upload-only --upload-to <dir>  # sync & verify only
python scripts/supervised_create_dataset.py --del-parsed         # ... after a verified upload
```

On Windows the original C++ supervisor is still available:

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

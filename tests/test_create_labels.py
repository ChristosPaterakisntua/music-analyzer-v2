"""Tests for the label creation and the dataset label handling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from music_analyzer_v2 import create_labels  # noqa: E402
from music_analyzer_v2.dataset import MusicDataset  # noqa: E402
from music_analyzer_v2.utils import FEELINGS, GENRES, SCALAR_FEATURE_NAMES  # noqa: E402


GENRE_TSV_HEADER = "track_id\tartist\talbum\tpath\taudio_duration\ttags"
MOOD_TSV_HEADER = "track_id\tartist\talbum\tpath\taudio_duration\ttags"


def write_split(directory: Path, split: str, genre_rows, mood_rows) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"autotagging_genre-{split}.tsv").write_text(
        "\n".join([GENRE_TSV_HEADER, *genre_rows]) + "\n", encoding="utf-8"
    )
    (directory / f"autotagging_moodtheme-{split}.tsv").write_text(
        "\n".join([MOOD_TSV_HEADER, *mood_rows]) + "\n", encoding="utf-8"
    )


@pytest.fixture()
def split_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data" / "splits" / "split-0"
    write_split(
        directory,
        "train",
        [
            "track_0000241\tartist\talbum\t41/241.mp3\t120\ttags\tgenre---pop\tgenre---rock",
            "track_00040148\tartist\talbum\t48/40148.mp3\t120\ttags\tgenre---jazz",
        ],
        [
            "track_0000241\tartist\talbum\t41/241.mp3\t120\ttags\tmood/theme---happy\tmood/theme---energetic",
            "track_00040148\tartist\talbum\t48/40148.mp3\t120\ttags\tmood/theme---sad",
        ],
    )
    return directory


def test_multi_label_rows_are_created(split_dir: Path, tmp_path: Path, monkeypatch) -> None:
    """A track with two genres and two moods must keep both labels."""
    monkeypatch.setattr(create_labels, "SPLITS_DIRECTORY", split_dir)
    monkeypatch.setattr(create_labels, "SPLITS", ("train",))

    entries = create_labels.create_entries_for_split("train")

    assert len(entries) == 2
    first = next(e for e in entries if e["id"] == "track_0000241")
    assert first["genres"] == ["pop", "rock"]
    assert first["feelings"] == ["happy", "energetic"]
    assert first["split"] == "train"


def test_label_audio_path_uses_the_low_quality_name(
    split_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(create_labels, "SPLITS_DIRECTORY", split_dir)
    monkeypatch.setattr(create_labels, "SPLITS", ("train",))

    entries = create_labels.create_entries_for_split("train")
    audios = {e["id"]: e["audio"] for e in entries}

    assert audios["track_0000241"].endswith("41/241.low.mp3")
    assert ".low.low." not in audios["track_0000241"]


def test_tracks_missing_in_one_subset_are_skipped(
    split_dir: Path, tmp_path: Path, monkeypatch
) -> None:
    # remove the mood row of the second track
    mood_file = split_dir / "autotagging_moodtheme-train.tsv"
    mood_file.write_text(
        "\n".join([MOOD_TSV_HEADER, "track_0000241\ta\tb\t41/241.mp3\t120\ttags\tmood/theme---happy"])
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(create_labels, "SPLITS_DIRECTORY", split_dir)

    entries = create_labels.create_entries_for_split("train")
    assert [e["id"] for e in entries] == ["track_0000241"]


def test_missing_split_file_is_treated_as_empty(tmp_path: Path) -> None:
    assert create_labels.read_split_file(tmp_path / "nope.tsv") == {}


def test_dataset_resolves_npz_files_next_to_the_labels_file(tmp_path: Path) -> None:
    """An overridden labels path must also redirect the .npz lookup."""
    labels_dir = tmp_path / "custom_parsed"
    labels_dir.mkdir()
    labels_path = labels_dir / "labels.json"

    labels_path.write_text(
        json.dumps(
            [
                {
                    "id": "track_0000241",
                    "audio": "/data/mtg-jamendo-dataset/dataset/41/241.low.mp3",
                    "genres": ["pop", "rock"],
                    "feelings": ["happy"],
                    "split": "train",
                }
            ]
        ),
        encoding="utf-8",
    )

    npz_path = labels_dir / "241.npz"
    np.savez(
        npz_path,
        log_mel_segments=np.zeros((2, 1, 64, 100), dtype=np.float32),
        scalar_features=np.ones(len(SCALAR_FEATURE_NAMES), dtype=np.float32),
    )

    dataset = MusicDataset(split="train", label_path=labels_path)
    assert len(dataset) == 1

    entry = dataset[0]
    assert entry.id == "track_0000241"
    # multi-hot targets: 2 genres and 1 feeling are active
    assert float(entry.genre_targets.sum()) == 2.0
    assert float(entry.feeling_targets.sum()) == 1.0
    assert entry.genre_targets.shape == (len(GENRES),)
    assert entry.feeling_targets.shape == (len(FEELINGS),)


def test_dataset_ignores_other_splits(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps(
            [
                {"id": "track_1", "audio": "1/1.low.mp3", "genres": ["pop"], "feelings": ["happy"], "split": "train"},
                {"id": "track_2", "audio": "2/2.low.mp3", "genres": ["rock"], "feelings": ["sad"], "split": "test"},
            ]
        ),
        encoding="utf-8",
    )

    dataset = MusicDataset(split="train", label_path=labels_path)
    assert [item["id"] for item in dataset.items] == ["track_1"]

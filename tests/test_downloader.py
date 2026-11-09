"""Tests for the MTG-Jamendo downloader target building."""

from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import download_mtg_jamendo as downloader  # noqa: E402


@pytest.fixture()
def manifest(tmp_path: Path) -> Path:
    """A labels.json with a track shared between two splits."""
    entries = [
        {
            "id": "track_0000241",
            "audio": "/data/mtg-jamendo-dataset/dataset/41/241.low.mp3",
            "genres": ["pop"],
            "feelings": ["happy"],
            "split": "train",
        },
        {
            # same track, different split -> must be downloaded only once
            "id": "track_0000241",
            "audio": "/data/mtg-jamendo-dataset/dataset/41/241.low.mp3",
            "genres": ["pop"],
            "feelings": ["happy"],
            "split": "test",
        },
        {
            "id": "track_00040148",
            "audio": "/data/mtg-jamendo-dataset/dataset/48/40148.low.mp3",
            "genres": ["rock"],
            "feelings": ["sad"],
            "split": "train",
        },
    ]
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_targets_use_the_universal_raw_30s_source(manifest: Path) -> None:
    targets = downloader.build_file_targets(manifest, "low", None)
    for url, _ in targets:
        assert url.startswith(
            "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/"
        ), url


def test_shared_tracks_are_downloaded_only_once(manifest: Path) -> None:
    targets = downloader.build_file_targets(manifest, "low", None)
    assert len(targets) == 2
    assert len({rel for _, rel in targets}) == 2


def test_relative_paths_are_inside_the_dataset_folder(manifest: Path) -> None:
    targets = downloader.build_file_targets(manifest, "low", None)
    relatives = sorted(rel.as_posix() for _, rel in targets)
    assert relatives == ["41/241.low.mp3", "48/40148.low.mp3"]


def test_quality_conversion_is_applied_to_the_tsv_path() -> None:
    entry = {"id": "track_0000241", "audio": "dataset/41/241.mp3"}
    assert downloader.rel_path_from_entry(entry, "low") == Path("41/241.low.mp3")
    assert downloader.rel_path_from_entry(entry, "full") == Path("41/241.mp3")


def test_quality_conversion_is_idempotent() -> None:
    entry = {"id": "track_0000241", "audio": "dataset/41/241.low.mp3"}
    assert downloader.rel_path_from_entry(entry, "low") == Path("41/241.low.mp3")


def test_multi_dot_names_do_not_get_a_doubled_suffix() -> None:
    entry = {"id": "track_0000241", "audio": "dataset/41/my.track.mp3"}
    result = downloader.rel_path_from_entry(entry, "low")
    assert result == Path("41/my.track.low.mp3")
    assert ".low.low." not in result.as_posix()


def test_fallback_to_the_track_id_without_an_audio_field() -> None:
    entry = {"id": "track_0000241"}
    assert downloader.rel_path_from_entry(entry, "low") == Path("41/241.low.mp3")


def test_full_quality_targets_use_the_audio_folder() -> None:
    entry = {"id": "track_0000241", "audio": "dataset/41/241.mp3"}
    rel = downloader.rel_path_from_entry(entry, "full")
    assert rel == Path("41/241.mp3")
    url = f"{downloader.CDN_BASE}/audio/{rel.as_posix()}"
    assert url == "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/41/241.mp3"


def test_limit_truncates_the_target_list(manifest: Path) -> None:
    targets = downloader.build_file_targets(manifest, "low", 1)
    assert len(targets) == 1


def test_manifest_path_can_be_a_plain_string(manifest: Path) -> None:
    """The manifest may be handed over as a string (e.g. from a CLI arg)."""
    targets = downloader.build_file_targets(str(manifest), "low", None)
    assert len(targets) == 2
    relatives = sorted(rel.as_posix() for _, rel in targets)
    assert relatives == ["41/241.low.mp3", "48/40148.low.mp3"]


def test_split_targets_point_at_the_official_metadata_repository() -> None:
    targets = downloader.build_split_targets()
    assert len(targets) == 6
    for url, rel in targets:
        assert url.startswith(
            "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/"
            "data/splits/split-0/"
        ), url
        assert rel.parts[:3] == ("data", "splits", "split-0")
        assert rel.name.startswith(("autotagging_genre-", "autotagging_moodtheme-"))
        assert rel.suffix == ".tsv"


def test_a_permanent_404_is_not_retried(tmp_path: Path, monkeypatch) -> None:
    """A 4xx will never succeed -> the downloader must give up immediately."""
    calls = {"count": 0}

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        raise urllib.error.HTTPError(
            request.full_url, 404, "Not Found", {}, None
        )

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fake_urlopen)

    status, _ = downloader.download_file(
        "https://example.invalid/x.mp3", tmp_path / "x.mp3", retries=5, timeout=1
    )

    assert status == "fail"
    assert calls["count"] == 1, "a 404 must not be retried"


def test_a_transient_503_is_retried(tmp_path: Path, monkeypatch) -> None:
    """A 5xx is transient (the CDN throttles) -> the downloader must retry."""
    calls = {"count": 0}
    real_sleep = downloader.time.sleep
    monkeypatch.setattr(downloader.time, "sleep", lambda *_: real_sleep(0))

    def fake_urlopen(request, timeout=None):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.HTTPError(
                request.full_url, 503, "Service Unavailable", {}, None
            )
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", fake_urlopen)

    status, _ = downloader.download_file(
        "https://example.invalid/x.mp3", tmp_path / "x.mp3", retries=5, timeout=1
    )

    assert status == "fail"
    assert calls["count"] == 3, "the 503 responses must be retried"

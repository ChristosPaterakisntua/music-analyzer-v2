"""Tests for the MTG-Jamendo download / label path helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (str(SRC_DIR), str(SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from music_analyzer_v2.utils import (  # noqa: E402
    low_quality_name,
    low_quality_path,
    relative_audio_path,
)


# --------------------------------------------------------------------------
# low_quality_name / low_quality_path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("241.mp3", "241.low.mp3"),
        ("241.low.mp3", "241.low.mp3"),  # idempotent
        ("40148.mp3", "40148.low.mp3"),
        ("my.track.name.mp3", "my.track.name.low.mp3"),
        ("my.track.name.low.mp3", "my.track.name.low.mp3"),
        ("no_extension", "no_extension.low.mp3"),
    ],
)
def test_low_quality_name(name: str, expected: str) -> None:
    assert low_quality_name(name) == expected


def test_low_quality_name_never_doubles_the_suffix() -> None:
    """`Path.with_suffix` would produce `241.low.low.mp3` — the bug we avoid."""
    once = low_quality_name("241.mp3")
    twice = low_quality_name(once)
    assert once == "241.low.mp3"
    assert twice == once


def test_low_quality_path_keeps_the_directory() -> None:
    assert low_quality_path(Path("41/241.mp3")) == Path("41/241.low.mp3")
    assert low_quality_path(Path("41/241.low.mp3")) == Path("41/241.low.mp3")


# --------------------------------------------------------------------------
# relative_audio_path
# --------------------------------------------------------------------------


def test_relative_audio_path_strips_the_dataset_marker() -> None:
    audio = "/data/mtg-jamendo-dataset/dataset/41/241.low.mp3"
    assert relative_audio_path(audio) == Path("41/241.low.mp3")


def test_relative_audio_path_uses_the_last_dataset_marker() -> None:
    """A parent folder named `dataset` must not hijack the result."""
    audio = "/mnt/dataset/backup/dataset/41/241.low.mp3"
    assert relative_audio_path(audio) == Path("41/241.low.mp3")


def test_relative_audio_path_windows_absolute_path() -> None:
    audio = r"C:\data\mtg-jamendo-dataset\dataset\41\241.low.mp3"
    assert relative_audio_path(audio) == Path("41/241.low.mp3")


def test_relative_audio_path_relative_input() -> None:
    assert relative_audio_path("dataset/41/241.mp3") == Path("41/241.mp3")
    assert relative_audio_path("41/241.mp3") == Path("41/241.mp3")


def test_relative_audio_path_without_marker_uses_last_two_parts() -> None:
    assert relative_audio_path("/somewhere/else/41/241.mp3") == Path("41/241.mp3")


def test_relative_audio_path_single_component() -> None:
    assert relative_audio_path("241.mp3") == Path("241.mp3")

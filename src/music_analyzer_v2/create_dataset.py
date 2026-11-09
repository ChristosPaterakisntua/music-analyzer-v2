from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from music_analyzer_v2.audio_loader import load_audio
from music_analyzer_v2.feature_extractor import ArrayExtractedFeatures, extract_features
from music_analyzer_v2.utils import (
    PARSED_DATASET_DIRECTORY,
    RAW_DATASET_DIRECTORY,
    check_valid_npz,
    get_path_from_id,
)


def manage_song(
    args: tuple[Path, Path, int],
) -> None:
    """
    Saves a `.npz` file with the extracted features of the audio.

    The file is named after the song id (track ids are unique), so it can be
    found later by :func:`music_analyzer_v2.utils.get_path_from_id`.

    Args
    ----
    args : `tuple[Path, Path, int]`
        ``(audio_path, output_dir, tries)``. The output directory is passed explicitly
        so that worker processes (which don't share the parent's globals after
        `spawn`) write to the right place. The function is executed at most `tries` times.

    Tracks whose `.npz` already exists are skipped, so reruns (e.g. after a
    crash) never redo or corrupt finished work.
    """
    path, output_dir, tries = args
    track_id = path.stem.removesuffix(".low")
    cache_path = output_dir / f"{track_id}.npz"

    if cache_path.exists():
        return
    try:
        audio_data = load_audio(path)
        features = ArrayExtractedFeatures(extract_features(audio_data))
        # `features` is a dataclass instance, so `**features` would raise:
        # unpack its `__dict__` instead
        np.savez_compressed(cache_path, **features.__dict__)
        if not check_valid_npz(
            track_id=track_id,
            delete_if_invalid=True,
            directory=output_dir,
        ):
            raise RuntimeError("Corrupt output")
    except Exception as error:
        if tries > 1:
            manage_song((args[0], args[1], tries - 1))
        else:
            print(error)


def create_parsed_dataset(
    batch_size: int = 5,
    num_workers: int | None = None,
    max_tracks: int | None = None,
    output_directory: Path | None = None,
    raw_directory: Path | None = None,
) -> None:
    """
    Used only for converting mtg-jamendo dataset to the custom parsed dataset

    Args
    ----
    batch_size : `int`
        number of tracks processed per batch
    num_workers : `int` | None
        number of processes to use. `None` uses all available CPUs.
    max_tracks : `int` | None
        optional cap on the number of tracks to process (useful for a quick
        test run). `None` processes the whole dataset.
    output_directory : `Path` | None
        optional output directory for the `.npz` files. `None` uses the
        default :data:`PARSED_DATASET_DIRECTORY`.
    raw_directory : `Path` | None
        optional directory that holds the raw `.low.mp3` audio. `None` uses
        the default :data:`RAW_DATASET_DIRECTORY`.
    """
    output_dir = (
        PARSED_DATASET_DIRECTORY if output_directory is None else Path(output_directory)
    )
    raw_dir = RAW_DATASET_DIRECTORY if raw_directory is None else Path(raw_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all audio files and keep only the ones that have not been parsed
    # yet (no matching `.npz`). This is robust against removed `.mp3` files and
    # crashes mid-batch: finished work is always skipped, whatever remains is
    # processed, so reruns never redo finished tracks or get stuck on an offset.
    paths: list[Path] = sorted(raw_dir.rglob("*.mp3"))

    pending: list[Path] = []
    for path in paths:
        track_id = path.stem.removesuffix(".low")
        # check inside the actual output dir (may be an overridden local dir)
        cache_path = get_path_from_id(track_id, directory=output_dir)
        if cache_path.exists():
            continue
        pending.append(path)

    if max_tracks is not None:
        pending = pending[:max_tracks]

    if max_tracks is not None:
        print(
            f"cap (max_tracks={max_tracks}) applied: processing {len(pending)} tracks."
        )
    elif pending:
        print(f"Processing {len(pending)} tracks that are not yet parsed.")
    if not pending:
        print("Nothing to do — every .mp3 already has a parsed .npz.")
        return

    total_batches: int = (len(pending) + batch_size - 1) // batch_size

    for batch_number, batch_start in enumerate(range(0, len(pending), batch_size)):
        batch = pending[batch_start : batch_start + batch_size]
        print(f"Batch {batch_number + 1}/{total_batches}")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # list(...) forces the tasks to finish and surfaces worker errors
            list(executor.map(manage_song, [(path, output_dir, 3) for path in batch]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse the raw MTG-Jamendo audio into .npz feature files"
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="where to write the .npz files (default: MA2_PARSED_DATASET)",
    )
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=None,
        help="where the raw .low.mp3 audio lives (default: MA2_RAW_DATASET/dataset)",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="process at most N tracks (testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="number of tracks per batch (default: 5)",
    )
    args = parser.parse_args()

    create_parsed_dataset(
        batch_size=args.batch_size,
        max_tracks=args.max_tracks,
        output_directory=args.output_directory,
        raw_directory=args.raw_directory,
    )

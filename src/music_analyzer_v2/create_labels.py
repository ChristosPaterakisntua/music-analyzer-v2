from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypedDict
import argparse

from music_analyzer_v2.utils import (
    FEELINGS,
    GENRES,
    LABELS_PATH,
    SPLITS_DIRECTORY,
    RAW_DATASET_DIRECTORY,
    PARSED_DATASET_DIRECTORY,
    check_valid_npz,
    low_quality_path,
)

GENRE_SOURCES: dict[str, set[str]] = {
    "classical": {
        "classical",
        "orchestral",
        "symphonic",
        "choir",
    },
    "soundtrack": {
        "soundtrack",
    },
    "ambient": {
        "ambient",
        "darkambient",
        "atmospheric",
        "newage",
    },
    "hip-hop": {
        "hiphop",
    },
    "pop": {
        "pop",
        "electropop",
        "synthpop",
        "poprock",
        "popfolk",
        "instrumentalpop",
    },
    "rock": {
        "rock",
        "alternativerock",
        "hardrock",
        "poprock",
        "instrumentalrock",
        "postrock",
        "grunge",
        "rocknroll",
        "classicrock",
        "bluesrock",
        "punkrock",
    },
    "jazz": {
        "jazz",
        "acidjazz",
        "jazzfusion",
        "swing",
        "bossanova",
    },
    "blues": {
        "blues",
        "bluesrock",
    },
    "rap": {
        "rap",
    },
    "R&B/soul": {
        "rnb",
        "soul",
    },
    "techno": {
        "techno",
    },
    "latin": {
        "latin",
        "bossanova",
    },
    "folk/acoustic": {
        "folk",
        "popfolk",
        "singersongwriter",
        "celtic",
        "country",
    },
    "metal": {
        "metal",
    },
    "reggae": {
        "reggae",
        "dub",
    },
    "electronic": {
        "electronic",
        "electronica",
        "ambient",
        "darkambient",
        "techno",
        "house",
        "deephouse",
        "trance",
        "dubstep",
        "drumnbass",
        "breakbeat",
        "triphop",
        "downtempo",
        "chillout",
        "idm",
        "industrial",
        "minimal",
        "synthpop",
        "electropop",
        "dance",
        "edm",
        "eurodance",
        "club",
        "darkwave",
        "newwave",
        "newage",
        "dub",
    },
    "dance/EDM": {
        "dance",
        "edm",
        "house",
        "deephouse",
        "trance",
        "dubstep",
        "drumnbass",
        "breakbeat",
        "eurodance",
        "club",
        "techno",
        "disco",
    },
    "punk": {
        "punkrock",
    },
}


FEELING_SOURCES: dict[str, set[str]] = {
    "happy": {"happy"},
    "uplifting": {"uplifting"},
    "energetic": {"energetic"},
    "calm": {"calm"},
    "relaxing": {"relaxing"},
    "dreamy": {"dream"},
    "romantic": {"romantic"},
    "nostalgic": set(),
    "melancholic": {"melancholic"},
    "sad": {"sad"},
    "dark": {"dark"},
    "tense": set(),
    "aggressive": set(),
    "epic": {"epic"},
}

# paths now come from music_analyzer_v2.utils (env-overridable, CWD-independent):
#   MTG_JAMENDO_DIRECTORY / SPLITS_DIRECTORY / RAW_DATASET_DIRECTORY / LABELS_PATH
AUDIO_DIRECTORY = RAW_DATASET_DIRECTORY

SPLITS = ("train", "validation", "test")


class TrackMetadata(TypedDict):
    path: str
    tags: list[str]


class LabelEntry(TypedDict):
    id: str
    audio: str
    genres: list[str]
    feelings: list[str]
    split: str


def read_split_file(path: Path) -> dict[str, TrackMetadata]:
    """Read an MTG-Jamendo TSV split file.

    A missing file is treated as an empty split (e.g. a checkout that only
    contains the train/test TSVs) instead of raising.
    """

    tracks: dict[str, TrackMetadata] = {}

    if not path.exists():
        print(f"WARNING: split file not found, treating it as empty: {path}")
        return tracks

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        try:
            next(reader)  # Skip header.
        except StopIteration:
            return tracks

        for row in reader:
            if len(row) < 6:
                continue

            track_id = row[0]
            relative_audio_path = row[3]
            tags = row[5:]

            tracks[track_id] = {
                "path": relative_audio_path,
                "tags": tags,
            }

    return tracks


def get_source_tags(
    tags: list[str],
    prefix: str,
) -> set[str]:
    """Remove a prefix such as 'genre---' from MTG tags."""

    return {tag.removeprefix(prefix) for tag in tags if tag.startswith(prefix)}


def map_tags(
    tags: list[str],
    prefix: str,
    target_order: list[str],
    mapping: dict[str, set[str]],
) -> list[str]:
    """Map MTG-Jamendo tags to the project's target labels."""

    source_tags = get_source_tags(tags, prefix)

    return [target for target in target_order if source_tags & mapping[target]]


def create_entries_for_split(
    split: str,
    check: bool = False,
    delete_if_invalid: bool = False,
    parsed_directory: Path | None = None,
) -> list[LabelEntry]:
    genre_file = SPLITS_DIRECTORY / f"autotagging_genre-{split}.tsv"
    mood_file = SPLITS_DIRECTORY / f"autotagging_moodtheme-{split}.tsv"

    genre_tracks = read_split_file(genre_file)
    mood_tracks = read_split_file(mood_file)

    # Keep only tracks that occur in both metadata subsets.
    common_track_ids = genre_tracks.keys() & mood_tracks.keys()

    # The .npz files live next to labels.json, so that an overridden
    # MA2_LABELS_PATH is honoured as well.
    npz_directory = parsed_directory if parsed_directory is not None else LABELS_PATH.parent

    entries: list[LabelEntry] = []

    for track_id in sorted(common_track_ids):
        if check and not check_valid_npz(
            track_id=track_id,
            delete_if_invalid=delete_if_invalid,
            directory=npz_directory,
        ):
            continue
        genre_metadata = genre_tracks[track_id]
        mood_metadata = mood_tracks[track_id]

        genres = map_tags(
            genre_metadata["tags"],
            prefix="genre---",
            target_order=GENRES,
            mapping=GENRE_SOURCES,
        )

        feelings = map_tags(
            mood_metadata["tags"],
            prefix="mood/theme---",
            target_order=FEELINGS,
            mapping=FEELING_SOURCES,
        )

        # Avoid ambiguous empty label lists.
        if not genres or not feelings:
            continue

        # The TSVs reference the full-quality files ("41/241.mp3"); the project
        # works on the low-quality ones ("41/241.low.mp3"). The conversion is
        # done on the string level so that multi-dot names are safe.
        audio_path = low_quality_path(AUDIO_DIRECTORY / genre_metadata["path"])

        entries.append(
            {
                "id": track_id,
                "audio": audio_path.as_posix(),
                "genres": genres,
                "feelings": feelings,
                "split": split,
            }
        )

    return entries


def main(
    check: bool = False,
    delete_if_invalid: bool = False,
) -> None:
    entries: list[LabelEntry] = []

    for split in SPLITS:
        split_entries = create_entries_for_split(
            split=split,
            check=check,
            delete_if_invalid=delete_if_invalid,
            parsed_directory=PARSED_DATASET_DIRECTORY,
        )
        entries.extend(split_entries)

        print(f"{split}: {len(split_entries)} tracks")

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            entries,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Total: {len(entries)} tracks")
    print(f"Created: {LABELS_PATH.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for corrupt .npz files & delete them"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="checks if the .npz files are correct to be added to labels.json",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete files (without this flag labels are created but files aren't deleted)",
    )
    args = parser.parse_args()
    main(check=args.check, delete_if_invalid=args.delete)

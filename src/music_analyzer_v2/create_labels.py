from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TypedDict
import argparse

from music_analyzer_v2.utils import FEELINGS, GENRES, check_valid_npz

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

RAW_DATASET_DIRECTORY = Path("../../mtg-jamendo-dataset")
SPLITS_DIRECTORY = RAW_DATASET_DIRECTORY / Path("data/splits/split-0")
AUDIO_DIRECTORY = RAW_DATASET_DIRECTORY / Path("dataset/")
OUTPUT_FILE = Path("../../parsed_dataset/labels.json")

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
    """Read an MTG-Jamendo TSV split file."""

    tracks: dict[str, TrackMetadata] = {}

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
) -> list[LabelEntry]:
    genre_file = SPLITS_DIRECTORY / f"autotagging_genre-{split}.tsv"
    mood_file = SPLITS_DIRECTORY / f"autotagging_moodtheme-{split}.tsv"

    genre_tracks = read_split_file(genre_file)
    mood_tracks = read_split_file(mood_file)

    # Keep only tracks that occur in both metadata subsets.
    common_track_ids = genre_tracks.keys() & mood_tracks.keys()

    entries: list[LabelEntry] = []

    for track_id in sorted(common_track_ids):
        if check and not check_valid_npz(
            track_id=track_id,
            delete_if_invalid=delete_if_invalid,
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

        audio_path = AUDIO_DIRECTORY / genre_metadata["path"]

        audio_path = audio_path.with_suffix(".low.mp3")

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
        )
        entries.extend(split_entries)

        print(f"{split}: {len(split_entries)} tracks")

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            entries,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Total: {len(entries)} tracks")
    print(f"Created: {OUTPUT_FILE.resolve()}")


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

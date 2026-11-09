"""
Deletes the .low.mp3 files whose features have already been parsed into .npz files.

Usage:
    python _delete_parsed_mp3.py           # dry-run (default): shows what WOULD be deleted
    python _delete_parsed_mp3.py --confirm # actually deletes the files
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# <repo>/scripts/<file>.py -> parents[0]=scripts, parents[1]=repo root
REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    """Reads a project path from an environment variable, else uses `default`.

    Mirrors the overrides of `music_analyzer_v2.utils` (MA2_RAW_DATASET /
    MA2_PARSED_DATASET) so this standalone script keeps working when the
    parsed dataset lives outside the repo (e.g. fast local disk on Colab).
    """
    value = os.environ.get(name)
    return Path(value) if value else default


RAW_DATASET_DIRECTORY = (
    _env_path("MA2_RAW_DATASET", REPO_ROOT / "mtg-jamendo-dataset") / "dataset"
)
PARSED_DATASET_DIRECTORY = _env_path(
    "MA2_PARSED_DATASET", REPO_ROOT / "parsed_dataset"
)


def remove_empty_dirs(root: Path) -> int:
    """Walk bottom-up and remove empty directories under ``root``."""
    removed = 0
    for dirpath, dirnames, filenames in os.walk(str(root), topdown=False):
        dir_path = Path(dirpath)
        if dir_path == root:
            continue
        try:
            if not dir_path.exists():
                continue
            if not any(dir_path.iterdir()):
                dir_path.rmdir()
                removed += 1
        except OSError:
            pass
    return removed


def main(confirm: bool = False) -> None:
    npz_files = sorted(PARSED_DATASET_DIRECTORY.glob("*.npz"))
    if not npz_files:
        print("No .npz files found in parsed_dataset/ — nothing to do.")
        return

    parsed_ids = {f.stem for f in npz_files}
    print(f"Found {len(parsed_ids)} parsed tracks in parsed_dataset/")

    # Build a map: track_id -> mp3 path
    mp3_map: dict[str, Path] = {}
    for mp3 in RAW_DATASET_DIRECTORY.rglob("*.low.mp3"):
        track_id = mp3.stem.replace(".low", "")
        if track_id in parsed_ids:
            mp3_map[track_id] = mp3

    missing = parsed_ids - mp3_map.keys()
    if missing:
        print(
            f"WARNING: {len(missing)} parsed tracks have no matching .mp3 "
            f"(already deleted or moved?)"
        )

    if not mp3_map:
        print("No matching .mp3 files to delete.")
        return

    total_size_mb = sum(p.stat().st_size for p in mp3_map.values()) / (1024 * 1024)
    print(f"\nWill delete {len(mp3_map)} .mp3 files ({total_size_mb:.1f} MB)")

    if not confirm:
        print("\n*** DRY RUN *** — no files were deleted.")
        print("Run with --confirm to actually delete.")
        return

    deleted = 0
    errors = 0
    for track_id, mp3_path in mp3_map.items():
        try:
            mp3_path.unlink()
            deleted += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR deleting {mp3_path}: {e}")
            errors += 1

    print(f"\nDone: {deleted} deleted, {errors} errors.")

    # Clean up empty directories left behind
    if deleted > 0:
        removed_dirs = remove_empty_dirs(RAW_DATASET_DIRECTORY)
        if removed_dirs:
            print(f"Cleaned up {removed_dirs} empty directories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delete .mp3 files that have already been parsed into .npz"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete files (without this flag, only a dry-run is shown)",
    )
    args = parser.parse_args()
    main(confirm=args.confirm)

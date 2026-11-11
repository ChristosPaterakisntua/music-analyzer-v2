"""
Downloads the MTG-Jamendo raw audio needed by this project into the repo.

The target file list is taken from ``labels.json`` (created by
``music_analyzer_v2.create_labels``), so only the tracks that will actually be
used are fetched. Downloads are parallel, resumable (Range-based continuation
of partial files) and verified against Content-Length.

Source of truth
---------------
All audio is fetched from the **universal** ``raw_30s`` subset of the official
CDN::

    https://cdn.freesound.org/mtg-jamendo/raw_30s/audio-low/<NN>/<id>.low.mp3

The per-subset paths (``autotagging_genre/audio-low/...``,
``autotagging_moodtheme/audio-low/...``) are **not** used: they return HTTP 404
for tracks that are not part of that specific subset, even though the exact same
file exists under ``raw_30s``. Since ``raw_30s`` contains the whole corpus it
covers every track of both the genre and the mood/theme splits, and tracks that
belong to both splits are therefore downloaded only once.

The official split TSVs reference the full-quality files (``41/241.mp3``);
the low-quality CDN serves the same audio as ``41/241.low.mp3``. The
conversion is done on the string level (see
:func:`music_analyzer_v2.utils.low_quality_name`) so that multi-dot file names
are handled correctly and the operation stays idempotent.

Usage:
    python scripts/download_mtg_jamendo.py --mode splits      # metadata TSVs only
    python scripts/download_mtg_jamendo.py                    # subset, low quality
    python scripts/download_mtg_jamendo.py --dry-run          # show what would be fetched
    python scripts/download_mtg_jamendo.py --limit 20         # quick test
    python scripts/download_mtg_jamendo.py --quality full     # 320 kbps stereo
    python scripts/download_mtg_jamendo.py --mode tars        # whole dataset as tar archives

The metadata split TSVs (``autotagging_genre-*.tsv`` /
``autotagging_moodtheme-*.tsv``) are **not** served by the CDN; they are
downloaded from the official GitHub repository into
``<raw dataset>/data/splits/split-0/`` (this is the path
:data:`music_analyzer_v2.utils.SPLITS_DIRECTORY` points to).

Environment overrides (same as music_analyzer_v2.utils):
    MA2_RAW_DATASET, MA2_PARSED_DATASET, MA2_LABELS_PATH
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# allow `import music_analyzer_v2.utils` from the src/ layout
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from music_analyzer_v2.utils import relative_audio_path  # noqa: E402

# <repo>/scripts/<file>.py -> parents[0]=scripts, parents[1]=repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
CDN_BASE = "https://cdn.freesound.org/mtg-jamendo/raw_30s"
# The split TSVs (metadata) are not served by the CDN - they live in the
# official GitHub repository.
SPLITS_BASE = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/splits/split-0"
)
SPLIT_FILES = (
    "autotagging_genre-train.tsv",
    "autotagging_genre-validation.tsv",
    "autotagging_genre-test.tsv",
    "autotagging_moodtheme-train.tsv",
    "autotagging_moodtheme-validation.tsv",
    "autotagging_moodtheme-test.tsv",
)
USER_AGENT = "music-analyzer-v2 downloader"
CHUNK_SIZE = 1 << 20  # 1 MiB
TAR_COUNT = 100  # raw_30s_audio[-low]-00.tar ... -99.tar


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def default_labels_path() -> Path:
    parsed = _env_path("MA2_PARSED_DATASET", REPO_ROOT / "parsed_dataset")
    return _env_path("MA2_LABELS_PATH", parsed / "labels.json")


def rel_path_from_entry(entry: dict, quality: str) -> Path:
    """Relative path (inside ``dataset/``) of one track's audio file.

    Prefers the ``audio`` field of ``labels.json`` (authoritative, comes from
    the MTG split TSVs) and falls back to deriving the folder from the track
    id (folder == last two digits of the id).

    The TSVs reference the full-quality files (``41/241.mp3``); the requested
    quality is applied by string manipulation so that names containing several
    dots are handled correctly (``241.mp3`` -> ``241.low.mp3`` and
    ``241.low.mp3`` -> ``241.low.mp3``, never ``241.low.low.mp3``).
    """
    audio = entry.get("audio") or ""
    suffix = ".low.mp3" if quality == "low" else ".mp3"

    rel_path = relative_audio_path(audio) if audio else None

    if rel_path is not None and len(rel_path.parts) >= 2:
        return rel_path.with_name(_apply_quality(rel_path.name, suffix))

    track_id = entry.get("id", "")
    number = int(re.sub("^track_0*", "", track_id))
    return Path(f"{number % 100:02d}") / f"{number}{suffix}"


def _apply_quality(name: str, suffix: str) -> str:
    """Returns ``name`` with the requested quality suffix applied."""
    for known in (".low.mp3", ".mp3"):
        if name.endswith(known):
            return name[: -len(known)] + suffix
    return name + suffix


def build_file_targets(labels_path: Path, quality: str, limit: int | None):
    """Builds the ``(url, relative path)`` list of every track in the manifest.

    Tracks that appear in several splits (e.g. the genre *and* the mood/theme
    split) resolve to the same relative path and are therefore downloaded only
    once.
    """
    with Path(labels_path).open("r", encoding="utf-8") as handle:
        entries = json.load(handle)

    subdir = "audio-low" if quality == "low" else "audio"
    seen: set[Path] = set()
    targets: list[tuple[str, Path]] = []
    for entry in entries:
        rel = rel_path_from_entry(entry, quality)
        if rel in seen:
            continue
        seen.add(rel)
        url = f"{CDN_BASE}/{subdir}/{rel.as_posix()}"
        targets.append((url, rel))

    if limit is not None:
        targets = targets[:limit]
    return targets


def build_split_targets() -> list[tuple[str, Path]]:
    """The split TSVs of the official metadata repository."""
    return [
        (f"{SPLITS_BASE}/{name}", Path("data") / "splits" / "split-0" / name)
        for name in SPLIT_FILES
    ]


def run_splits_mode(dest_root: Path, retries: int, dry_run: bool) -> int:
    """Downloads the metadata split TSVs (needed to build the labels)."""
    targets = build_split_targets()
    print(f"Metadata: {len(targets)} split files")
    if dry_run:
        for url, rel in targets:
            print(f"  would fetch {url} -> {dest_root / rel}")
        return 0

    failures: list[str] = []
    for url, rel in targets:
        dest = dest_root / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  [skip] {rel.name}")
            continue
        status, _ = download_file(url, dest, retries=retries, timeout=120)
        if status == "fail":
            failures.append(url)
        else:
            print(f"  [{status}] {rel.name}")

    if failures:
        print(f"FAILED: {len(failures)} metadata files")
        for url in failures:
            print(f"  {url}")
        return 1
    print("Metadata ready.")
    return 0


def build_tar_targets(quality: str) -> list[tuple[str, Path]]:
    prefix = "raw_30s_audio-low" if quality == "low" else "raw_30s_audio"
    return [
        (f"{CDN_BASE}/{prefix}-{index:02d}.tar", Path("_tars") / f"{prefix}-{index:02d}.tar")
        for index in range(TAR_COUNT)
    ]


def _retry_backoff(error: urllib.error.HTTPError, attempt: int) -> float:
    """Backoff in seconds for a throttling (5xx) response.

    Honours a ``Retry-After`` header when the server sends one; otherwise falls
    back to a jittered exponential backoff so parallel workers don't retry in
    lock-step.
    """
    retry_after = error.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return max(1, int(retry_after))
        except ValueError:
            pass
    return min(15 * (2 ** (attempt - 1)), 300) * random.uniform(0.75, 1.25)


def download_file(url: str, dest: Path, retries: int, timeout: int = 120) -> tuple[str, int]:
    """Downloads one file with resume support.

    Returns ``(status, bytes_downloaded)`` where status is one of
    ``skip`` / ``ok`` / ``fail``.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return "skip", 0

    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                resume_ok = offset > 0 and response.status == 206
                if offset > 0 and not resume_ok:
                    # server ignored the Range header: restart from scratch
                    offset = 0
                expected_header = response.headers.get("Content-Length")
                expected_total = (
                    offset + int(expected_header) if expected_header is not None else None
                )
                mode = "ab" if resume_ok else "wb"
                written = offset
                with part.open(mode) as handle:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        written += len(chunk)

            if expected_total is not None and written != expected_total:
                raise IOError(f"incomplete: {written} != {expected_total} bytes")

            os.replace(part, dest)
            return "ok", written - offset
        except urllib.error.HTTPError as error:
            # 404 (and 4xx in general) will never succeed -> do not waste retries
            if error.code < 500:
                print(f"    [{error.code}] {url} -> not available")
                if part.exists():
                    part.unlink()
                return "fail", 0
            # 5xx (the CDN throttles with 503 under load) -> retry with a long,
            # jittered exponential backoff; the file is definitely there.
            if attempt == retries:
                if part.exists():
                    part.unlink()
                return "fail", 0
            backoff = _retry_backoff(error, attempt)
            print(f"    [{error.code}] throttled, retrying in {backoff:.0f}s "
                  f"({attempt}/{retries})")
            time.sleep(backoff)
        except Exception:  # noqa: BLE001 - network errors of all kinds
            if attempt == retries:
                if part.exists():
                    part.unlink()  # corrupted partial, drop it
                return "fail", 0
            time.sleep(min(2 ** attempt, 30) * random.uniform(0.75, 1.25))
    return "fail", 0


def _progress(done: int, total: int, downloaded_files: int, bytes_done: int, started: float):
    elapsed = max(time.time() - started, 1e-6)
    rate_mb = bytes_done / (1 << 20) / elapsed
    eta = (total - done) * elapsed / done if done else 0
    print(
        f"  {done}/{total} files ({downloaded_files} new, "
        f"{bytes_done / (1 << 30):.2f} GiB, {rate_mb:.1f} MiB/s, ETA {eta / 60:.0f} min)",
        flush=True,
    )


def run_files_mode(targets, dest_root: Path, workers: int, retries: int, dry_run: bool) -> int:
    destinations = [dest_root / "dataset" / rel for _, rel in targets]
    missing = [
        (url, path)
        for (url, _), path in zip(targets, destinations)
        if not (path.exists() and path.stat().st_size > 0)
    ]
    print(
        f"Tracks in manifest: {len(targets)} | already present: "
        f"{len(targets) - len(missing)} | to download: {len(missing)}"
    )
    if dry_run:
        for url, path in missing[:20]:
            print(f"  would fetch {url} -> {path}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        return 0
    if not missing:
        print("Nothing to do — audio already complete.")
        return 0

    started = time.time()
    done = 0
    new_files = 0
    bytes_done = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_file, url, path, retries): url
            for url, path in missing
        }
        for future in as_completed(futures):
            url = futures[future]
            status, size = future.result()
            done += 1
            if status == "ok":
                new_files += 1
                bytes_done += size
            elif status == "fail":
                failures.append(url)
            if done % 200 == 0 or done == len(missing):
                _progress(done, len(missing), new_files, bytes_done, started)

    if failures:
        print(f"FAILED: {len(failures)} files (rerun the script to retry):")
        for url in failures[:10]:
            print(f"  {url}")
        return 1
    print(f"Done: {new_files} new files, {bytes_done / (1 << 30):.2f} GiB.")
    return 0


def run_tars_mode(dest_root: Path, quality: str, retries: int, dry_run: bool) -> int:
    """Downloads the 100 tar archives of the *whole* dataset and extracts them."""
    targets = build_tar_targets(quality)
    print(
        f"Tar mode ({quality}): {len(targets)} archives "
        f"({'~165 GiB' if quality == 'low' else '~540 GiB'} total)."
    )
    if dry_run:
        for url, rel in targets[:5]:
            print(f"  would fetch {url} -> {dest_root / rel}")
        print("  ...")
        return 0

    for index, (url, rel) in enumerate(targets, start=1):
        tar_path = dest_root / rel
        print(f"[{index}/{len(targets)}] {rel.name}", flush=True)
        status, _ = download_file(url, tar_path, retries=retries, timeout=600)
        if status == "fail":
            print(f"FAILED to download {url} — rerun to resume.")
            return 1
        # extract (archive root contains the 00/..99/ folders, sometimes prefixed)
        with tarfile.open(tar_path) as archive:
            first = archive.getmembers()[0].name
            target = dest_root if first.startswith("dataset/") else dest_root / "dataset"
            try:
                archive.extractall(path=target, filter="data")
            except TypeError:  # Python without the tarfile filter parameter
                archive.extractall(path=target)
        tar_path.unlink()  # extracted: free the space
    print("Tar mode done.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Downloads MTG-Jamendo raw audio")
    parser.add_argument("--labels", type=Path, default=None,
                        help="labels.json manifest (default: MA2_LABELS_PATH or parsed_dataset/labels.json)")
    parser.add_argument("--dest", type=Path, default=None,
                        help="MTG-Jamendo repo root (default: MA2_RAW_DATASET or <repo>/mtg-jamendo-dataset)")
    parser.add_argument("--quality", choices=("low", "full"), default="low",
                        help="low = mono ~100 kbps .low.mp3 (recommended); full = 320 kbps stereo")
    parser.add_argument("--mode", choices=("files", "tars", "splits"), default="files",
                        help="files = only tracks from labels.json; tars = whole dataset archives; "
                             "splits = only the metadata split TSVs")
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads (default 4)")
    parser.add_argument("--retries", type=int, default=10, help="attempts per file (default 10)")
    parser.add_argument("--limit", type=int, default=None, help="download at most N files (testing)")
    parser.add_argument("--dry-run", action="store_true", help="only print what would be done")
    args = parser.parse_args()

    labels_path = args.labels or default_labels_path()
    dest_root = args.dest or _env_path("MA2_RAW_DATASET", REPO_ROOT / "mtg-jamendo-dataset")

    if args.mode == "tars":
        return run_tars_mode(dest_root, args.quality, args.retries, args.dry_run)

    if args.mode == "splits":
        return run_splits_mode(dest_root, args.retries, args.dry_run)

    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found.\n"
              "Run `python -m music_analyzer_v2.create_labels` first to create the manifest.")
        return 1

    targets = build_file_targets(labels_path, args.quality, args.limit)
    if not targets:
        print("ERROR: labels.json contains no tracks.")
        return 1
    return run_files_mode(targets, dest_root, args.workers, args.retries, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())

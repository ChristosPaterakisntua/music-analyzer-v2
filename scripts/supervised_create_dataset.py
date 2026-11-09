"""
Portable (Windows/Linux) watchdog around `music_analyzer_v2.create_dataset`.

- Keeps re-running the dataset creation until it finishes cleanly with no new
  work (same logic as the old `supervised_create_dataset.cpp`, but counting the
  *actual* output directory instead of assuming the repo layout).
- Optionally uploads every newly created `.npz` (and `labels.json`) to a target
  directory — typically a Google Drive mount — after each successful round,
  tracked by a size manifest so re-runs are idempotent.
- `--del-parsed` deletes the already-parsed `.mp3` files ONLY after every local
  `.npz` has been verified present (same size) in the upload target.

Usage:
    python scripts/supervised_create_dataset.py
    python scripts/supervised_create_dataset.py --upload-to /content/drive/MyDrive/music-analyzer_v2/parsed_dataset
    python scripts/supervised_create_dataset.py --upload-only          # just sync & verify, don't parse
    python scripts/supervised_create_dataset.py --del-parsed           # after a verified upload
    python scripts/supervised_create_dataset.py --dry-run

Environment overrides (same as music_analyzer_v2.utils):
    MA2_RAW_DATASET, MA2_PARSED_DATASET, MA2_LABELS_PATH
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# <repo>/scripts/<file>.py -> parents[0]=scripts, parents[1]=repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DELETE_MP3_SCRIPT = REPO_ROOT / "scripts" / "delete_parsed_mp3.py"
MANIFEST_NAME = ".uploaded_manifest.json"


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


def default_parsed_dir() -> Path:
    return _env_path("MA2_PARSED_DATASET", REPO_ROOT / "parsed_dataset")


def default_labels_path(parsed_dir: Path) -> Path:
    return _env_path("MA2_LABELS_PATH", parsed_dir / "labels.json")


def count_npz(directory: Path) -> int:
    try:
        return sum(1 for path in directory.glob("*.npz") if path.is_file())
    except OSError:
        return 0


def child_env() -> dict[str, str]:
    """Environment for child processes: package importable via src/ layout."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing if existing else "")
    return env


def run_create_dataset() -> int:
    command = [sys.executable, "-m", "music_analyzer_v2.create_dataset"]
    completed = subprocess.run(command, cwd=str(REPO_ROOT), env=child_env())
    return completed.returncode


def load_manifest(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_manifest(path: Path, manifest: dict[str, int]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def upload_new(
    parsed_dir: Path,
    upload_dir: Path,
    labels_path: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Copies new/changed `.npz` (+ labels.json) to `upload_dir`.

    A size manifest inside `parsed_dir` makes the sync idempotent; every file
    is staged as `.part` and moved in place only after its size was verified,
    so an interrupted upload never corrupts the target.

    Returns ``(uploaded_files, uploaded_bytes)``.
    """
    manifest_path = parsed_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)

    candidates: list[Path] = sorted(parsed_dir.glob("*.npz"))
    if labels_path.exists():
        candidates.append(labels_path)

    pending = []
    for source in candidates:
        try:
            size = source.stat().st_size
        except OSError:
            continue
        if manifest.get(source.name) == size:
            continue
        pending.append((source, size))

    if not pending:
        return 0, 0
    if dry_run:
        for source, size in pending[:10]:
            print(f"  would upload {source.name} ({size} bytes)")
        if len(pending) > 10:
            print(f"  ... and {len(pending) - 10} more")
        return 0, 0

    upload_dir.mkdir(parents=True, exist_ok=True)
    uploaded = 0
    uploaded_bytes = 0
    for source, size in pending:
        target = upload_dir / source.name
        staged = upload_dir / (source.name + ".part")
        try:
            shutil.copy2(source, staged)
            if staged.stat().st_size != size:
                raise IOError(f"size mismatch after copy of {source.name}")
            os.replace(staged, target)
        except OSError as error:
            if staged.exists():
                staged.unlink()
            print(f"  upload failed for {source.name}: {error}")
            continue
        manifest[source.name] = size
        uploaded += 1
        uploaded_bytes += size

    save_manifest(manifest_path, manifest)
    return uploaded, uploaded_bytes


def verify_uploads(parsed_dir: Path, upload_dir: Path) -> tuple[int, int]:
    """Checks every local `.npz` exists in `upload_dir` with the same size.

    Returns ``(verified, missing)``.
    """
    verified = 0
    missing = 0
    for source in sorted(parsed_dir.glob("*.npz")):
        target = upload_dir / source.name
        try:
            if target.exists() and target.stat().st_size == source.stat().st_size:
                verified += 1
                continue
        except OSError:
            pass
        missing += 1
        if missing <= 10:
            print(f"  MISSING on target: {source.name}")
    return verified, missing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run create_dataset until done, with optional Drive upload"
    )
    parser.add_argument("--parsed-dir", type=Path, default=None,
                        help="local dir holding the .npz files "
                             "(default: MA2_PARSED_DATASET or <repo>/parsed_dataset)")
    parser.add_argument("--upload-to", type=Path, default=None,
                        help="target dir (e.g. Google Drive mount) that receives the new .npz files")
    parser.add_argument("--upload-only", action="store_true",
                        help="skip parsing: one sync + verification round, then exit")
    parser.add_argument("--max-no-progress", type=int, default=3,
                        help="abort after N restarts without new .npz (default 3)")
    parser.add_argument("--del-parsed", action="store_true",
                        help="after DONE and a verified upload, delete parsed .mp3 files")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would run / be uploaded, don't do it")
    args = parser.parse_args()

    parsed_dir = (args.parsed_dir or default_parsed_dir()).resolve()
    labels_path = default_labels_path(parsed_dir)
    upload_dir = args.upload_to.resolve() if args.upload_to else None
    parsed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[watch] parsed dir : {parsed_dir}")
    print(f"[watch] labels     : {labels_path}")
    print(f"[watch] upload to  : {upload_dir or '(none — local only)'}")

    if args.upload_only:
        uploaded, uploaded_bytes = upload_new(
            parsed_dir, upload_dir or parsed_dir, labels_path, dry_run=args.dry_run
        )
        if upload_dir and not args.dry_run:
            verified, missing = verify_uploads(parsed_dir, upload_dir)
            print(f"[watch] verify: {verified} ok, {missing} missing")
            return 0 if missing == 0 else 1
        print(f"[watch] uploaded {uploaded} files ({uploaded_bytes / (1 << 20):.1f} MiB)")
        return 0

    no_progress = 0
    try:
        while True:
            before = count_npz(parsed_dir)
            print(f"[watch] .npz now: {before} -> running create_dataset...", flush=True)
            if args.dry_run:
                print("[watch] dry-run: would run `python -m music_analyzer_v2.create_dataset`")
                return 0
            return_code = run_create_dataset()
            after = count_npz(parsed_dir)

            if upload_dir:
                uploaded, uploaded_bytes = upload_new(parsed_dir, upload_dir, labels_path)
                if uploaded:
                    print(f"[watch] uploaded {uploaded} files "
                          f"({uploaded_bytes / (1 << 20):.1f} MiB) to {upload_dir}")

            if return_code == 0 and after == before:
                print(f"[watch] DONE: {after} .npz files, nothing left to do.")
                if upload_dir:
                    verified, missing = verify_uploads(parsed_dir, upload_dir)
                    print(f"[watch] final verify: {verified} ok, {missing} missing")
                    if missing:
                        print("[watch] NOT deleting mp3s — upload incomplete.")
                        return 1
                    if args.del_parsed:
                        completed = subprocess.run(
                            [sys.executable, str(DELETE_MP3_SCRIPT), "--confirm"],
                            env=child_env(),
                        )
                        print(f"[watch] delete_parsed_mp3 exit code: {completed.returncode}")
                elif args.del_parsed:
                    print("[watch] --del-parsed refused without --upload-to "
                          "(mp3s are the only copy of the source audio).")
                    return 1
                return 0

            if after > before:
                print(f"[watch] progress {before} -> {after}, restarting...")
                no_progress = 0
                continue

            no_progress += 1
            if no_progress > args.max_no_progress:
                print(f"[watch] STUCK: no progress after {args.max_no_progress} restarts.")
                return 1
            print(f"[watch] no progress (exit {return_code}), "
                  f"restart {no_progress}/{args.max_no_progress}")
    except KeyboardInterrupt:
        print("\n[watch] interrupted — rerun the same command to resume "
              "(finished tracks are skipped).")
        return 130


if __name__ == "__main__":
    sys.exit(main())

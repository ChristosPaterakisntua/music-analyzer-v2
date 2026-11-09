import random
import re
from pathlib import Path
import zipfile
import os

import numpy as np
import torch
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============= DATA TYPES ============

FloatArray = NDArray[np.float32]
FloatingArray = NDArray[np.floating]


# =============== DATA ================

GENRES = [
    "classical",
    "soundtrack",
    "ambient",
    "hip-hop",
    "pop",
    "rock",
    "jazz",
    "blues",
    "rap",
    "R&B/soul",
    "techno",
    "latin",
    "folk/acoustic",
    "metal",
    "reggae",
    "electronic",
    "dance/EDM",
    "punk",
]

FEELINGS = [
    "happy",
    "uplifting",
    "energetic",
    "calm",
    "relaxing",
    "dreamy",
    "romantic",
    "nostalgic",
    "melancholic",
    "sad",
    "dark",
    "tense",
    "aggressive",
    "epic",
]

STATISTIC_NAMES = [
    "mean",
    "min",
    "max",
    "std",
    "median",
    "mean_delta",
]

ARRAY_FEATURE_NAMES = [
    "rms_energy",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
]

SCALAR_FEATURE_NAMES = [
    *[
        f"{feature_name}_{statistic_name}"
        for feature_name in ARRAY_FEATURE_NAMES
        for statistic_name in STATISTIC_NAMES
    ],
    "duration",
    "harmonic_percussive_ratio",
    "dynamic_range",
    "key_sin",
    "key_cos",
    "mode",  # 0 -> minor, 1 -> major
    "key_mode_match_score",
    "tempo",
]

SEED = 42
GENRE_THRESHOLD = 0.5
FEELING_THRESHOLD = 0.5


# ================ PATHS =================

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    """Reads a project path from an environment variable, else uses `default`.

    This keeps the local workflow untouched (defaults resolve exactly like the
    old CWD-relative paths) while allowing e.g. a Colab run to point the raw /
    parsed dataset to a Google-Drive mount or a fast local disk — from any
    working directory.
    """
    value = os.environ.get(name)
    return Path(value) if value else default


# Overridable with environment variables (useful on Colab / network mounts):
#   MA2_RAW_DATASET        -> MTG-Jamendo repo root (contains data/ and dataset/)
#   MA2_PARSED_DATASET     -> output dir of the parsed .npz files (+ labels.json)
#   MA2_LABELS_PATH        -> labels.json location (defaults inside parsed dir)
#   MA2_CHECKPOINT_DIR     -> training checkpoints
#   MA2_SAVED_MODELS_DIR   -> final trained models
MTG_JAMENDO_DIRECTORY = _env_path(
    "MA2_RAW_DATASET", REPO_ROOT / "mtg-jamendo-dataset"
)
# raw audio (`.low.mp3` files) inside the MTG-Jamendo repo
RAW_DATASET_DIRECTORY = MTG_JAMENDO_DIRECTORY / "dataset"
SPLITS_DIRECTORY = MTG_JAMENDO_DIRECTORY / "data" / "splits" / "split-0"
PARSED_DATASET_DIRECTORY = _env_path(
    "MA2_PARSED_DATASET", REPO_ROOT / "parsed_dataset"
)
LABELS_PATH = _env_path("MA2_LABELS_PATH", PARSED_DATASET_DIRECTORY / "labels.json")
CHECKPOINT_DIR = _env_path("MA2_CHECKPOINT_DIR", REPO_ROOT / "checkpoints")
SAVED_MODELS_DIR = _env_path("MA2_SAVED_MODELS_DIR", REPO_ROOT / "saved_models")


# ============== FUNCTIONS ================


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    if torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def ask_yes_or_no(prompt: str) -> bool:
    answers = ["y", "n"]
    prompt = prompt.strip() + " (y/n): "
    ans = ""
    while ans not in answers:
        ans = input(prompt).lower().strip()
    return ans == "y"


def multi_hot(
    labels: list[str],
    classes: list[str],
    label_type: str,
) -> FloatArray:
    """
    Creates a multi-hot-vector from the given labels according to classes
    e.g. :data:`music_analyzer_v2.create_labels.GENRES`

    The output array has zeros and ones but for better compatibility
    the dtype is :class:`np.float32`

    Args
    -------
    labels : list[str]
        audio's labels
        e.g. `["pop", "rock"]`
    classes : list[str]
        format of this label type
        e.g. :data:`music_analyzer_v2.create_labels.GENRES`
    label_type : str
        label type
        e.g. `"genre"`
    """
    class_to_index = {class_name: index for index, class_name in enumerate(classes)}

    unknown = sorted(set(labels) - set(class_to_index.keys()))
    if unknown:
        raise ValueError(f"Unknown {label_type} labels: {', '.join(unknown)}")

    multi_hot_array = np.zeros(len(classes), dtype=np.float32)
    for label in labels:
        multi_hot_array[class_to_index[label]] = 1.0

    return multi_hot_array


def low_quality_name(name: str) -> str:
    """Returns the ``.low.mp3`` filename of a raw MTG-Jamendo audio filename.

    The official split TSVs reference the full-quality files (``41/241.mp3``)
    while the low-quality CDN serves ``41/241.low.mp3``. The transformation is
    done on the *string* level (not with :meth:`pathlib.Path.with_suffix`, which
    would happily turn ``241.low.mp3`` into ``241.low.low.mp3``) so that it is
    idempotent and safe for names containing several dots.

    Args
    ----
    name : `str`
        the file name (not the full path), e.g. ``"241.mp3"``.
    """
    for suffix in (".low.mp3", ".mp3"):
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".low.mp3"
    return name + ".low.mp3"


def low_quality_path(path: Path) -> Path:
    """Same as :func:`low_quality_name` but keeps the directory structure."""
    return path.with_name(low_quality_name(path.name))


def relative_audio_path(audio: str | Path) -> Path:
    """Extracts the path of a track *inside* the MTG-Jamendo ``dataset/`` folder.

    ``labels.json`` stores the ``audio`` field as the path that
    :mod:`music_analyzer_v2.create_labels` built, i.e.
    ``<raw dataset>/dataset/<NN>/<id>.low.mp3``. The raw dataset root may live
    anywhere (it is env-overridable), so the absolute prefix has to be stripped
    before the path can be used as a download target.

    The ``dataset/`` marker is looked up as a *path component* (and the last
    occurrence wins), so a parent directory that happens to be named
    ``dataset`` cannot hijack the result — unlike a naive
    ``audio.index("dataset/")``.

    Args
    ----
    audio : `str` | `Path`
        the ``audio`` value of a label entry.

    Returns
    -------
    `Path`
        e.g. ``Path("41/241.low.mp3")``.
    """
    path = Path(audio)
    parts = path.parts

    # last occurrence of a "dataset" component
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "dataset":
            return Path(*parts[index + 1 :]) if index + 1 < len(parts) else Path()

    # fallback: MTG-Jamendo always stores the audio as "<NN>/<id>.mp3"
    if len(parts) >= 2:
        return Path(*parts[-2:])

    return path


def get_path_from_id(id: str, directory: Path | None = None) -> Path:
    """returns the path pointing to the parsed dataset given the audio id

    Args
    ----
    id : `str`
        track id (e.g. `track_0000241`)
    directory : `Path` | None
        the parsed dataset directory. `None` uses
        :data:`PARSED_DATASET_DIRECTORY` (env-overridable via
        `MA2_PARSED_DATASET`).
    """
    path = Path(re.sub("^track_0*", "", id) + ".npz")
    base = PARSED_DATASET_DIRECTORY if directory is None else Path(directory)
    return base / path


def validate_inputs(
    log_mel_segments: torch.Tensor,
    scalar_features: torch.Tensor,
) -> None:
    if log_mel_segments.ndim != 5:
        raise ValueError(
            "log_mel_segments must have shape "
            "(batch, segments, channels, n_mels, frames)"
        )

    if scalar_features.ndim != 2:
        raise ValueError("scalar_features must have shape (batch, scalar_dim)")

    batch_size = log_mel_segments.shape[0]
    channels = log_mel_segments.shape[2]

    if channels != 1:
        raise ValueError("Expected one log-mel input channel")

    if scalar_features.shape[0] != batch_size:
        raise ValueError("The scalar and log-mel batch sizes differ")


def check_valid_npz(
    track_id: str,
    delete_if_invalid: bool = False,
    directory: Path | None = None,
) -> bool:
    path: Path = get_path_from_id(track_id, directory=directory)
    res: bool = True
    try:
        if not zipfile.is_zipfile(path):
            res = False
    except (FileNotFoundError, zipfile.BadZipFile):
        res = False
    if res:
        try:
            # A parsed file holds ONE song, so log_mel has no batch dimension:
            # (segments, 1, n_mels, frames). validate_inputs expects a batch, so
            # add it with unsqueeze(0) → (1, segments, 1, n_mels, frames).
            with np.load(path) as data:
                log_mel_segments = torch.from_numpy(data["log_mel_segments"]).unsqueeze(
                    0
                )
                scalar_features = torch.from_numpy(data["scalar_features"]).unsqueeze(0)

                validate_inputs(
                    log_mel_segments=log_mel_segments,
                    scalar_features=scalar_features,
                )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            res = False
    if not res:
        if delete_if_invalid:
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
        else:
            print(f"Broken npz {path}")
    return res


def create_generator(seed: int = SEED) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def set_random_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logits_to_predictions(
    logits: torch.Tensor,
    classes: list[str],
    threshold: float,
) -> list[str]:
    probabilities = np.squeeze(np.array(torch.sigmoid(logits)))
    if probabilities.ndim != 1:
        raise ValueError(
            "Expected a single prediction (shape (num_classes,) or (1, num_classes))"
        )
    mask = probabilities >= threshold
    return np.array(classes)[mask].tolist()


def outputs_to_predictions(
    outputs: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    """
    Converts model outputs to predictions in user friendly format

    Returns
    --------
    tuple[list[str], list[str]] : (genres, feelings)
    """
    genres = logits_to_predictions(
        logits=outputs["genre_logits"],
        classes=GENRES,
        threshold=GENRE_THRESHOLD,
    )
    feelings = logits_to_predictions(
        logits=outputs["feeling_logits"],
        classes=FEELINGS,
        threshold=FEELING_THRESHOLD,
    )
    return genres, feelings


def micro_f1(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> float:
    """
    Calculates the micro f1 score

    Args
    -------
    true_positives : int
        The model predicted 1 when the true label was 1.
        e.g. prediction for genre: rock: 1, expected: rock: 1
    false_positives : int
        The model predicted 1 when the true label was 0.
        e.g. prediction for genre: rock: 1, expected: rock: 0
    false_negatives : int
        The model predicted 0 when the true label was 1.
        e.g. prediction for genre: rock: 0, expected: rock: 1

    We don't use true negatives (TN) because this would make the model
    think it did a good job when it didn't

    Key concepts
    ------------
    TP: true positives

    FP: false positives

    FN: false negatives

    Precision = TP / (TP + FP)
        how many of the ones where accurate?

    Recall = TP / (TP + FN)
        how many of the ones did the model find? (also known as sensitivity)

    micro_f1 = harmonic_mean(Precision, Recall) = (2 * TP) / (2 * TP + FP + FN)
        We need to take both precision and recall into account to
        evaluate the model's accuracy for multi-label classification
        tasks
    """
    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return 0.0
    return 2.0 * true_positives / denominator


def get_multilabel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
) -> tuple[int, int, int, int]:
    """
    Calculates true_positives, false_positives, false_negatives
    from prediction logits

    Used for metrics (not training)

    Returns
    --------
    tuple[int, int, int, int]
        true_positives, false_positives, false_negatives, true_negatives
    """
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= threshold
    expected = targets >= 0.5

    true_positives = int(torch.logical_and(predictions, expected).sum().item())
    false_positives = int(torch.logical_and(predictions, ~expected).sum().item())
    false_negatives = int(torch.logical_and(~predictions, expected).sum().item())
    true_negatives = int(torch.logical_and(~probabilities, ~expected).sum().item())

    return true_positives, false_positives, false_negatives, true_negatives


def create_confusion_matrix(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    true_negatives: int,
    title: str = "",
) -> None:
    """Constructs the confusion matrix used for evaluating the actual model accuracy"""
    values = np.array(
        [
            [true_positives, false_negatives],
            [false_positives, true_negatives],
        ]
    )
    labels = np.array(
        [
            ["TP", "FN"],
            ["FP", "TN"],
        ]
    )
    colors = np.array(
        [
            ["lightgreen", "red"],
            ["red", "lightgreen"],
        ]
    )
    _, ax = plt.subplots()

    for i in range(2):
        for j in range(2):
            rect = Rectangle((j, i), 1, 1, facecolor=colors[i, j], edgecolor="black")
            ax.add_patch(rect)
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{labels[i, j]}\n{values[i, j]}",
                ha="center",
                va="center",
                fontsize=12,
            )
    ax.set_xticks([0.5, 1.5], ["Positive", "Negative"])
    ax.set_yticks([0.5, 1.5], ["Positive", "Negative"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{title} Confusion Matrix")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

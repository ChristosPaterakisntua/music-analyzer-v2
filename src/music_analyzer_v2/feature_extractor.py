from dataclasses import dataclass

import librosa
import numpy as np
from numpy.typing import NDArray

from .audio_loader import AudioData
from .utils import SCALAR_FEATURE_NAMES, FloatArray, FloatingArray

PROPERTIES_OF_INTEREST: list[str] = [
    "duration",
    "tempo",
    "rms_energy",
    "dynamic_range",
    "zero_crossing_rate",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_contrast",
    "MFCC",
    "chroma",
    "key",
    "mode",
    "harmonic_percussive_ratio",
    "log_mel_segments",
    "sequence_segments",
]

SEGMENT_SECONDS = 5.0
MIN_SEGMENT_FRACTION = 0.5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
N_MFCC = 20

PITCH_CLASSES = np.array(
    [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]
)

# Krumhansl–Kessler key profiles
MAJOR_PROFILE = np.array(
    [
        6.35,
        2.23,
        3.48,
        2.33,
        4.38,
        4.09,
        2.52,
        5.19,
        2.39,
        3.66,
        2.29,
        2.88,
    ]
)

MINOR_PROFILE = np.array(
    [
        6.33,
        2.68,
        3.52,
        5.38,
        2.60,
        3.53,
        2.54,
        4.75,
        3.98,
        2.69,
        3.34,
        3.17,
    ]
)

# Its indices are stored in ExtractedFeatures.scalar_features
MODES: list[str] = [
    "minor",
    "major",
]


@dataclass
class FeatureStatistics:
    """Holds mean, min, max, std values of a given value"""

    name: str
    mean: float
    minimum: float
    maximum: float
    std: float
    median: float
    mean_delta: float

    def __init__(self, name: str, data: NDArray):
        self.name = name
        self.mean = float(np.mean(data))
        self.minimum = float(np.min(data))
        self.maximum = float(np.max(data))
        self.std = float(np.std(data))
        self.median = float(np.median(data))
        delta: NDArray = np.diff(data, axis=-1)
        self.mean_delta = float(np.mean(np.abs(delta))) if delta.size > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        """Creates a dictionary with all the properties except name"""
        res: dict[str, float] = {}
        res[f"{self.name}_mean"] = self.mean
        res[f"{self.name}_min"] = self.minimum
        res[f"{self.name}_max"] = self.maximum
        res[f"{self.name}_std"] = self.std
        res[f"{self.name}_median"] = self.median
        res[f"{self.name}_mean_delta"] = self.mean_delta
        return res


def _add_statistics(features: dict[str, float], name: str, data: NDArray) -> None:
    statistics = FeatureStatistics(name, data)
    features.update(statistics.to_dict())


@dataclass
class ExtractedFeatures:
    scalar_features: dict[str, float]
    # Shape: (segments, 1, n_mels, frames)
    log_mel_segments: FloatArray
    # Shape: (segments, feature_channels, frames)
    sequence_segments: FloatArray


@dataclass
class ArrayExtractedFeatures:
    """
    Same as :class:`ExtractedFeatures` but the scalar features are
    stored in a :class:`FloatArray`

    Easier to save to a `.npz` file and convert to tensors
    """

    # Shape: (38, )
    scalar_features: FloatArray
    # Shape: (segments, 1, n_mels, frames)
    log_mel_segments: FloatArray
    # Shape: (segments, feature_channels, frames)
    sequence_segments: FloatArray

    def __init__(self, extracted_features: ExtractedFeatures):
        # checks
        missing_features = [
            name
            for name in SCALAR_FEATURE_NAMES
            if name not in extracted_features.scalar_features
        ]
        if missing_features:
            raise ValueError("Missing scalar features: " + ", ".join(missing_features))

        # order matters
        values = [
            extracted_features.scalar_features[feature]
            for feature in SCALAR_FEATURE_NAMES
        ]

        if not np.all(np.isfinite(values)):
            raise ValueError("Scalar features contain NaN or infinity")

        self.scalar_features = np.array(values, dtype=np.float32)
        self.log_mel_segments = extracted_features.log_mel_segments
        self.sequence_segments = extracted_features.sequence_segments


def _split_into_segments(
    y: FloatingArray,
    sr: int,
) -> list[FloatingArray]:
    """
    Splits a :data:`FloatingArray` cutting every :data:`SEGMENT_SECONDS`
    """
    if y.size == 0:
        raise ValueError("Audio contains no samples")
    segments: list[FloatingArray] = []
    segment_samples = round((SEGMENT_SECONDS * sr))
    for start in range(0, len(y), segment_samples):
        segment = y[start : start + segment_samples]
        fraction = len(segment) / segment_samples
        if fraction < MIN_SEGMENT_FRACTION:
            break
        segment = librosa.util.fix_length(segment, size=segment_samples)
        segments.append(segment)
    return segments


def _extract_segment_matrices(
    segment: FloatingArray,
    sr: int,
) -> tuple[FloatArray, FloatArray]:
    # 1. MEL spectrogram: shape (128, frames)
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    # Conversion from power to db
    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    ).astype(np.float32)

    # 2. MFCC: shape (20, frames)
    mfcc = librosa.feature.mfcc(
        S=log_mel,
        n_mfcc=N_MFCC,
    ).astype(np.float32)

    # 3. Chroma: shape (12, frames)
    # tuning=0.0 skips librosa's internal tuning estimation, which can hard-crash
    # (numba access violation) on segments with little tonal content.
    chroma = librosa.feature.chroma_cens(
        y=segment,
        sr=sr,
        hop_length=HOP_LENGTH,
        tuning=0.0,
    ).astype(np.float32)

    # 4. Spectral contrast: usual shape (7, frames)
    spectral_contrast = librosa.feature.spectral_contrast(
        y=segment,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    ).astype(np.float32)

    # Make sure all matrices have the same number of time frames
    target_frames = log_mel.shape[-1]
    mfcc = librosa.util.fix_length(
        mfcc,
        size=target_frames,
        axis=-1,
    )
    chroma = librosa.util.fix_length(
        chroma,
        size=target_frames,
        axis=-1,
    )
    spectral_contrast = librosa.util.fix_length(
        spectral_contrast,
        size=target_frames,
        axis=-1,
    )

    # shape: (39, frames)
    sequence_features = np.concatenate(
        [mfcc, chroma, spectral_contrast], axis=0
    ).astype(np.float32)

    return log_mel, sequence_features


def _get_correlation(values: FloatingArray, template: FloatingArray) -> float:
    """Returns Pearson correlation between 12 value profiles"""
    return float(np.corrcoef(values, template)[0, 1])


def _estimate_key_mode(
    harmonic: FloatingArray,
    sr: int,
) -> tuple[str, str, float]:
    """Estimate global key, mode and uncalibrated match score"""
    # tuning=0.0 skips librosa's internal tuning estimation, which can hard-crash
    # (numba access violation) on tracks with little tonal content.
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=sr,
        hop_length=HOP_LENGTH,
        tuning=0.0,
    )
    average_chroma = np.mean(chroma, axis=1)
    if not np.all(np.isfinite(average_chroma)):
        raise ValueError("Chroma contains invalid values")

    if np.std(average_chroma) < 1e-8:
        raise ValueError("Cannot estimate key from silent or tonally ambiguous audio")

    scores: list[tuple[float, int, str]] = []
    for tonic_index in range(len(PITCH_CLASSES)):
        major_template = np.roll(MAJOR_PROFILE, tonic_index)
        minor_template = np.roll(MINOR_PROFILE, tonic_index)
        scores.append(
            (
                _get_correlation(average_chroma, major_template),
                tonic_index,
                "major",
            )
        )
        scores.append(
            (
                _get_correlation(average_chroma, minor_template),
                tonic_index,
                "minor",
            )
        )

    best_score, key_index, mode = max(
        scores,
        key=lambda result: result[0],
    )
    key = str(PITCH_CLASSES[key_index])
    return key, mode, best_score


def _encode_key_mode(
    key: str,
    mode: str,
) -> tuple[float, float, float]:
    """Encode key cyclically and mode as an integer value"""
    key_indices = np.where(PITCH_CLASSES == key)[0]
    if key_indices.size == 0:
        raise ValueError(f"Invalid key: {key}")
    key_index = int(key_indices[0])
    angle = 2.0 * np.pi * key_index / len(PITCH_CLASSES)
    key_sin = float(np.sin(angle))
    key_cos = float(np.cos(angle))
    imode = float(MODES.index(mode))
    return key_sin, key_cos, imode


def extract_features(audio: AudioData) -> ExtractedFeatures:
    """
    Extracts the audio's properties of interest

    See also
    ---------
    :data:`PROPERTIES_OF_INTEREST`,
    :class:`AudioData`,
    :class:`ExtractedFeatures`
    """

    y = np.asarray(audio.samples)
    sr = audio.sample_rate

    if y.ndim != 1:
        raise ValueError("The feature extractor currently expects mono audio")

    scalar_features: dict[str, float] = {}

    # Single (NDArray) features
    rms_energy = librosa.feature.rms(
        y=y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]
    spectral_centroid_matrix = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    spectral_centroid = spectral_centroid_matrix[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        centroid=spectral_centroid_matrix,
    )[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )[0]

    array_features: dict[str, NDArray] = {
        "rms_energy": rms_energy,
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
    }

    for name, data in array_features.items():
        _add_statistics(scalar_features, name, data)

    # Simple (float) features
    duration = audio.duration_seconds
    scalar_features["duration"] = duration

    harmonic, percussive = librosa.effects.hpss(y=y)
    harmonic_energy = float(np.mean(harmonic**2))
    percussive_energy = float(np.mean(percussive**2))
    epsilon = 1e-10
    harmonic_percussive_ratio = float(
        10.0 * np.log10((harmonic_energy + epsilon) / (percussive_energy + epsilon))
    )
    scalar_features["harmonic_percussive_ratio"] = harmonic_percussive_ratio

    rms_db = librosa.amplitude_to_db(
        rms_energy,
        ref=np.max,
        top_db=None,
    )
    dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
    scalar_features["dynamic_range"] = dynamic_range

    key, mode, key_match_score = _estimate_key_mode(harmonic, sr)
    key_sin, key_cos, imode = _encode_key_mode(
        key,
        mode,
    )
    scalar_features["key_sin"] = key_sin
    scalar_features["key_cos"] = key_cos
    scalar_features["mode"] = imode
    scalar_features["key_mode_match_score"] = key_match_score

    # scalar feature
    tempo = librosa.feature.tempo(
        y=y,
        sr=sr,
        hop_length=HOP_LENGTH,
    )[0]
    scalar_features["tempo"] = float(tempo)

    # matrix features
    segments = _split_into_segments(y, sr)
    if not segments:
        raise ValueError("Audio is too short to produce any segments")

    log_mel_segments: list[FloatArray] = []
    sequence_segments: list[FloatArray] = []

    for segment in segments:
        log_mel, sequence = _extract_segment_matrices(
            segment,
            sr,
        )
        # 2D CNN
        log_mel_segments.append(log_mel[np.newaxis, :, :])
        # 1D CNN
        sequence_segments.append(sequence)

    return ExtractedFeatures(
        scalar_features=scalar_features,
        log_mel_segments=np.stack(log_mel_segments),
        sequence_segments=np.stack(sequence_segments),
    )


def validate_extracted_features(
    result: ExtractedFeatures,
) -> None:
    expected_sequence_channels = N_MFCC + 12 + 7

    if result.log_mel_segments.ndim != 4:
        raise ValueError("log_mel_segments must have 4 dimensions")

    if result.sequence_segments.ndim != 3:
        raise ValueError("sequence_segments must have 3 dimensions")

    if result.log_mel_segments.shape[1] != 1:
        raise ValueError("log_mel_segments must have one CNN channel")

    if result.log_mel_segments.shape[2] != N_MELS:
        raise ValueError(f"Expected {N_MELS} Mel bands")

    if result.sequence_segments.shape[1] != (expected_sequence_channels):
        raise ValueError("Unexpected number of sequence channels")

    if not np.all(np.isfinite(result.log_mel_segments)):
        raise ValueError("log_mel_segments contains NaN or infinity")

    if not np.all(np.isfinite(result.sequence_segments)):
        raise ValueError("sequence_segments contains NaN or infinity")

    if not all(np.isfinite(value) for value in result.scalar_features.values()):
        raise ValueError("scalar_features contains NaN or infinity")

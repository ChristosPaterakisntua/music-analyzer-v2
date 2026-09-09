from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .utils import FloatArray

DEFAULT_SAMPLE_RATE = 22_050
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
MONO = True


class AudioLoadError(RuntimeError):
    """Error during the loading of the audio file"""


@dataclass
class AudioData:
    """
    The custom data structure returned by the loader

    Properties
    ----------
    samples : `FloatArray`
        The audio's waveform in one-dimension numpy array format
    sample_rate : `int`
        The number of audio samples (measurements) per second
    duration_seconds : `float`
        The audio's duration in seconds
    path : `Path`
        The audio's full path

    See also
    -------
    :data:`music_analyzer_v2.utils.FloatArray`
    """

    samples: FloatArray
    sample_rate: int
    duration_seconds: float
    path: Path

    @property
    def number_of_samples(self) -> int:
        return len(self.samples)


def load_audio(
    file_path: str | Path,
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> AudioData:
    """
    Loads the audio file using librosa with given sample rate,
    default: :data:`DEFAULT_SAMPLE_RATE`

    Args
    -------
    file_path : `str` | `Path`
        path to the audio file
    target_sample_rate : `int`
        rate of samples (measurements per second) used for loading

    Returns
    --------
    AudioData with waveform, sample rate, duration in seconds and path

    Raises
    --------
    AudioLoadError:
        In case the file doesn't exist or it is not supported or
        cannot be loaded
    """
    path = Path(file_path)
    _validate_path(path)
    _validate_sample_rate(target_sample_rate)
    try:
        samples, sample_rate = librosa.load(
            path=path,
            sr=target_sample_rate,
            mono=MONO,
            dtype=np.float32,
        )
    except Exception as error:
        raise AudioLoadError(f"Failed to load path {path}. Details: {error}") from error

    if samples.size == 0:
        raise AudioLoadError(f"{path} has no sound data")

    if not np.all(np.isfinite(samples)):
        raise AudioLoadError(f"{path} has invalid sound values")

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    duration_seconds = len(samples) / sample_rate

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        path=path,
    )


def _validate_path(path: Path) -> None:
    """Checks if the given path corresponds to a supported audio file"""
    if not path.exists():
        raise AudioLoadError(f"File not found: {path}")

    if not path.is_file():
        raise AudioLoadError(f"{path} is not a file")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise AudioLoadError(
            f"Not supported audio type: {path.suffix or 'no suffix'}. "
            f"Supported: {supported}"
        )


def _validate_sample_rate(sample_rate: int) -> None:
    """Checks if sample rate is positive integer"""
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
        raise AudioLoadError("Sample rate must be an int")

    if sample_rate <= 0:
        raise AudioLoadError("Sample rate must be positive")

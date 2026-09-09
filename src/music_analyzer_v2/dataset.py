from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import (
    FEELINGS,
    GENRES,
    FloatArray,
    create_generator,
    get_path_from_id,
    multi_hot,
)

LABELS_PATH = Path("../../parsed_dataset/labels.json")

NUM_WORKERS = 0
BATCH_SIZE = 4


@dataclass
class AudioEntry:
    """
    Properties
    ----------
    log_mel_segments : `torch.Tensor`
        mel spectogram
    scalar_features : `torch.Tensor`
        shape: (len(:data:`.utils.SCALAR_FEATURE_NAMES`),)
    genre_targets : `torch.Tensor`
        multi hot array
    feeling_targets : `torch.Tensor`
        multi hot array
    id : `str`
    """

    # mel spectogram
    log_mel_segments: torch.Tensor
    # shape (len(SCALAR_FEATURE_NAMES),)
    scalar_features: torch.Tensor
    # multi hot array
    genre_targets: torch.Tensor
    # multi hot array
    feeling_targets: torch.Tensor
    id: str


class MusicDataset(Dataset[dict[str, torch.Tensor]]):
    """
    Loads "songs" from the parsed dataset.
    In reality it loads the extracted features from `.npz` files.

    Inputs
    --------
    split : `str`
        should be any of `["train", "validation", "test"]`
    label_path : `Path`
        default: :data:`LABELS_PATH`

    Note
    ------
    Use :method:`fit_scalar_normalizer` and :method:`set_scalar_normalizer`
    """

    def __init__(
        self,
        split: str,
        label_path: Path = LABELS_PATH,
    ) -> None:
        splits: list[str] = ["train", "validation", "test"]
        if split not in splits:
            raise ValueError(f"Valid split categories are: {', '.join(splits)}")

        self.label_path = label_path

        with open(label_path, "r") as f:
            raw_items: list[dict[str, Any]] = json.load(f)

        self.items: list[dict[str, Any]] = [
            item for item in raw_items if item.get("split") == split
        ]

        # will be later used for scaling the scalar features
        self.scalar_mean: FloatArray | None = None
        self.scalar_std: FloatArray | None = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int) -> AudioEntry:
        """
        See also: :class:`AudioEntry`
        """
        item = self.items[index]
        track_id = item.get("id")
        path = get_path_from_id(track_id)
        features = np.load(path)
        genres = list(item.get("genres"))
        feelings = list(item.get("feelings"))
        genre_targets = multi_hot(
            labels=genres,
            classes=GENRES,
            label_type="genre",
        )
        feeling_targets = multi_hot(
            labels=feelings, classes=FEELINGS, label_type="feeling"
        )
        scalar_features = features.get("scalar_features")

        if self.scalar_mean is not None and self.scalar_std is not None:
            scalar_features = (
                (scalar_features - self.scalar_mean) / self.scalar_std
            ).astype(np.float32)

        return AudioEntry(
            log_mel_segments=torch.from_numpy(features.get("log_mel_segments")),
            scalar_features=torch.from_numpy(scalar_features),
            genre_targets=torch.from_numpy(genre_targets),
            feeling_targets=torch.from_numpy(feeling_targets),
            id=track_id,
        )

    def fit_scalar_normalizer(self) -> tuple[FloatArray, FloatArray]:
        """
        Use it only for train split

        Returns
        ---------
        tuple[FloatArray, FloatArray] :
            (mean, std)
        """
        scalar_values: list[FloatArray] = []
        for idx in range(len(self)):
            tensor = self[idx].scalar_features
            array = tensor.cpu().detach().numpy()
            scalar_values.append(np.array(array, dtype=np.float32))
        matrix = np.stack(scalar_values, axis=0).astype(np.float32)
        mean = np.mean(matrix, axis=0).astype(np.float32)
        std = np.std(matrix, axis=0).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        return mean, std

    def set_scalar_normalizer(self, mean: FloatArray, std: FloatArray) -> None:
        self.scalar_mean = mean
        self.scalar_std = std


def collate_songs(batch: list[AudioEntry]) -> dict[str, Any]:
    """
    Collate function for the songs by adding padding to log mel
    tensors in order to `torch.stack` them

    Returns
    ----------
    dict[str, Any]
        - log_mel_segments (padded)
        - scalar_features (stacked)
        - segment_lengths
        - genre_targets (stacked)
        - feeling_targets (stacked)
        - ids (list of str, not a tensor)
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")

    segment_lengths = torch.tensor(
        [sample.log_mel_segments.shape[0] for sample in batch],
        dtype=torch.long,
    )

    max_segments = int(segment_lengths.max())

    first_shape = batch[0].log_mel_segments.shape[1:]
    channels, n_mels, frames = first_shape

    padded_log_mel = torch.zeros(
        (
            len(batch),
            max_segments,
            channels,
            n_mels,
            frames,
        ),
        dtype=torch.float32,
    )

    for idx, sample in enumerate(batch):
        log_mel = sample.log_mel_segments
        if log_mel.shape[1:] != first_shape:
            raise ValueError(
                "All log-mel segments need to have the same "
                "channel, Mel, frame dimensions",
            )
        number_of_segments = log_mel.shape[0]
        padded_log_mel[idx, :number_of_segments] = log_mel

    return {
        "log_mel_segments": padded_log_mel,
        "scalar_features": torch.stack([sample.scalar_features for sample in batch]),
        "segment_lengths": segment_lengths,
        "genre_targets": torch.stack([sample.genre_targets for sample in batch]),
        "feeling_targets": torch.stack([sample.feeling_targets for sample in batch]),
        "ids": [sample.id for sample in batch],
    }


class MusicDataLoader(DataLoader):
    """
    The custom dataloader for :class:`MusicDataset`.
    Works the same way as a regular torch :class:`DataLoader`.

    Inputs
    --------
    dataset : `MusicDataset`
    device : `torch.device`
    batch_size : `int`
        default: :data:`BATCH_SIZE`
    shuffle : `bool`
        default: `True`
    num_workers : `int`
        default: :data:`NUM_WORKERS`

    Note
    -----
    Use `shuffle=False` for test / validation
    """

    def __init__(
        self,
        dataset: MusicDataset,
        device: torch.device,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
        num_workers: int = NUM_WORKERS,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_songs,
            pin_memory=device.type == "cuda",
            generator=create_generator(),
        )

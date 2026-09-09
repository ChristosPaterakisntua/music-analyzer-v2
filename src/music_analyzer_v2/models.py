from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from .utils import FEELINGS, GENRES, SCALAR_FEATURE_NAMES, FloatArray

CONV_KERNEL_SIZE = 3


# Helper classes (ml blocks)


class ConvBlock(nn.Module):
    """
    Basic CNN block:

    Conv2D -> BatchNorm -> GELU ->
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=CONV_KERNEL_SIZE,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels,
            ),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(
                in_features=in_dim,
                out_features=128,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                in_features=128,
                out_features=embed_dim,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegmentCNN(nn.Module):
    """
    Converts one log-Mel segment into a fixed-size embedding.

    Input:
        (batch_segments, 1, n_mels, frames)

    Output:
        (batch_segments, embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            ConvBlock(
                in_channels=1,
                out_channels=32,
                dropout=0.05,
            ),
            ConvBlock(
                in_channels=32,
                out_channels=64,
                dropout=0.1,
            ),
            ConvBlock(
                in_channels=64,
                out_channels=128,
                dropout=0.15,
            ),
            ConvBlock(in_channels=128, out_channels=256, dropout=0.2),
            # Whatever the spectogram dimensions are,
            # the result becomes (256, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        return self.projection(x)


class MusicAnalyzerModel0(nn.Module):
    """
    Double input, double output music classification model

    Inputs
    -----
    log_mel_segments:
        Shape: (batch, max_segments, 1, n_mels, frames)

    segment_lengths:
        Number of real segments for each song.
        Shape: (batch,)

    scalar_features:
        Shape: (len(:data:`.utils.SCALAR_FEATURE_NAMES`),)

    Outputs
    ------
    genre_logits:
        Shape: (batch, num_genres)

    feeling_logits:
        Shape: (batch, num_feelings)
    """

    def __init__(
        self,
        scalar_dim: int = len(SCALAR_FEATURE_NAMES),
        num_genres: int = len(GENRES),
        num_feelings: int = len(FEELINGS),
        segment_embedding_dim: int = 256,
        gru_hidden_size: int = 128,
        scalar_embedding_dim: int = 64,
        shared_hidden_size: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if scalar_dim <= 0:
            raise ValueError("scalar_dim must be positive")

        if num_genres <= 1:
            raise ValueError("num_genres must be greater than 1")

        if num_feelings <= 0:
            raise ValueError("num_feelings must be positive")

        self.scalar_dim = scalar_dim

        # 1. CNN: one embedding for every five-second segment.
        self.segment_encoder = SegmentCNN(
            embedding_dim=segment_embedding_dim,
            dropout=0.2,
        )

        # 2. BiGRU: combines the segments in their original order.
        self.segment_gru = nn.GRU(
            input_size=segment_embedding_dim,
            hidden_size=gru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        # The BiGRU has two directions.
        song_embedding_dim = 2 * gru_hidden_size

        # 3. MLP for scalar features
        self.scalar_encoder = MLPBlock(
            in_dim=scalar_dim,
            embed_dim=scalar_embedding_dim,
            dropout=dropout,
        )

        fusion_dim = song_embedding_dim + scalar_embedding_dim

        # 4. Commmon represantation for both tasks.
        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features=fusion_dim,
                out_features=shared_hidden_size,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(shared_hidden_size),
        )

        # 5. Seperate prediction heads.
        self.genre_head = nn.Linear(
            shared_hidden_size,
            num_genres,
        )

        self.feeling_head = nn.Linear(
            shared_hidden_size,
            num_feelings,
        )

    def forward(
        self,
        log_mel_segments: torch.Tensor,
        segment_lengths: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(
            log_mel_segments=log_mel_segments,
            segment_lengths=segment_lengths,
            scalar_features=scalar_features,
        )

        (
            batch_size,
            max_segments,
            channels,
            n_mels,
            frames,
        ) = log_mel_segments.shape

        # Merge batch and segment dimensions
        #
        # (B, S, 1, 128, T)
        #        ↓
        # (B*S, 1, 128, T)
        flat_segments = log_mel_segments.reshape(
            batch_size * max_segments,
            channels,
            n_mels,
            frames,
        )

        # CNN embedding for every segment:
        #
        # (B*S, 1, 128, T)
        #       ↓
        # (B*S, segment_embedding_dim)
        segment_embeddings = self.segment_encoder(flat_segments)

        # Restore song and segment dimensions:
        #
        # (B*S, embedding)
        #       ↓
        # (B, S, embedding)
        segment_embeddings = segment_embeddings.reshape(
            batch_size,
            max_segments,
            -1,
        )

        # Ignores padded segments when songs have
        # different number of segments
        packed_segments = pack_padded_sequence(
            input=segment_embeddings,
            lengths=segment_lengths.to(
                device="cpu",
                dtype=torch.long,
            ),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.segment_gru(packed_segments)

        # For a one-layer bidirectional GRU:
        #
        # hidden[-2] = final forward state
        # hidden[-1] = final backward state
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        scalar_embedding = self.scalar_encoder(scalar_features)

        song_embedding = torch.cat(
            [
                forward_hidden,
                backward_hidden,
            ],
            dim=1,
        )

        combined_embedding = torch.cat(
            [
                song_embedding,
                scalar_embedding,
            ],
            dim=1,
        )

        shared_embedding = self.shared_layers(combined_embedding)
        genre_logits = self.genre_head(shared_embedding)
        feeling_logits = self.feeling_head(shared_embedding)

        return {
            "genre_logits": genre_logits,
            "feeling_logits": feeling_logits,
        }

    def _validate_inputs(
        self,
        log_mel_segments: torch.Tensor,
        segment_lengths: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> None:
        if log_mel_segments.ndim != 5:
            raise ValueError(
                "log_mel_segments must have shape "
                "(batch, segments, channels, n_mels, frames)"
            )

        if scalar_features.ndim != 2:
            raise ValueError("scalar_features must have shape (batch, scalar_dim)")

        if segment_lengths.ndim != 1:
            raise ValueError("segment lengths must have shape (batch,)")

        batch_size = log_mel_segments.shape[0]
        max_segments = log_mel_segments.shape[1]
        channels = log_mel_segments.shape[2]

        if channels != 1:
            raise ValueError("Expected one log-mel input channel")

        if scalar_features.shape[0] != batch_size:
            raise ValueError("The scalar and log-mel batch sizes differ")

        if segment_lengths.shape[0] != batch_size:
            raise ValueError("segment_lengths has the wrong batch size")

        if scalar_features.shape[1] != self.scalar_dim:
            raise ValueError(
                f"Expected {self.scalar_dim} got {scalar_features.shape[1]}"
            )

        lengths = segment_lengths.detach().cpu()

        if torch.any(lengths < 1):
            raise ValueError("Every song must contain at least one segment")

        if torch.any(lengths > max_segments):
            raise ValueError("A segment length exceeds max_segments")


# Generic type hint for all valid models
type MusicAnalyzerModel = type[MusicAnalyzerModel0]


def load_saved_model(
    path: str | Path,
    device: torch.device,
) -> tuple[MusicAnalyzerModel0, FloatArray | None, FloatArray | None]:
    """
    Loads a saved model plus, when available, the scalar normalizer that was
    fitted on the train split.

    Returns
    -------
    tuple[MusicAnalyzerModel0, FloatArray | None, FloatArray | None]
        (model, scalar_mean, scalar_std)

    Note
    ----
    Checkpoints saved without the normalizer (e.g. older runs) return
    `None` for both `scalar_mean` and `scalar_std`.
    """
    saved_data = torch.load(Path(path), map_location=device, weights_only=False)
    model = MusicAnalyzerModel0()
    model.load_state_dict(saved_data["model_state_dict"])
    model.to(device).eval()

    scalar_mean = saved_data.get("scalar_mean")
    scalar_std = saved_data.get("scalar_std")

    return model, scalar_mean, scalar_std

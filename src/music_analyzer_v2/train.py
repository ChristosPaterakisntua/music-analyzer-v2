from __future__ import annotations

from pathlib import Path
from shutil import copy
from typing import Any

import torch
from torch import nn

from .dataset import MusicDataLoader
from .models import MusicAnalyzerModel0
from .utils import (
    FEELING_THRESHOLD,
    GENRE_THRESHOLD,
    FloatArray,
    get_multilabel_metrics,
    micro_f1,
    set_random_seed,
)

EPOCHS = 50
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
FEELING_LOSS_WEIGHT = 1.0
MAX_GRADIENT_NORM = 5.0

CHECKPOINT_DIR = Path("../../checkpoints")
SAVED_MODELS_DIR = Path("../../saved_models")


CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def run_epoch(
    model: MusicAnalyzerModel0,
    data_loader: MusicDataLoader,
    device: torch.device,
    genre_loss_function: nn.Module,
    feeling_loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    """
    Performs an epoch

    Returns
    --------
    dict[str, float] :
        Keys:
        - loss
        - genre_loss
        - feeling_loss
        - genre_micro_f1
        - feeling_micro_f1
    """

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_genre_loss = 0.0
    total_feeling_loss = 0.0
    total_songs = 0

    genre_tp = genre_fp = genre_fn = 0
    feeling_tp = feeling_fp = feeling_fn = 0

    for batch in data_loader:
        log_mel_segments = batch["log_mel_segments"].to(
            device,
            non_blocking=True,
        )
        scalar_features = batch["scalar_features"].to(
            device,
            non_blocking=True,
        )
        segment_lengths = batch["segment_lengths"].to(
            device,
            non_blocking=True,
        )
        genre_targets = batch["genre_targets"].to(
            device,
            non_blocking=True,
        )
        feeling_targets = batch["feeling_targets"].to(
            device,
            non_blocking=True,
        )

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(
                log_mel_segments=log_mel_segments,
                segment_lengths=segment_lengths,
                scalar_features=scalar_features,
            )

            genre_logits = outputs["genre_logits"]
            feeling_logits = outputs["feeling_logits"]

            genre_loss = genre_loss_function(
                genre_logits,
                genre_targets,
            )
            feeling_loss = feeling_loss_function(
                feeling_logits,
                feeling_targets,
            )
            loss = genre_loss + FEELING_LOSS_WEIGHT * feeling_loss

            if is_training:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    parameters=model.parameters(),
                    max_norm=MAX_GRADIENT_NORM,
                )
                optimizer.step()

        batch_size = log_mel_segments.shape[0]
        total_songs += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_genre_loss += float(genre_loss.detach()) * batch_size
        total_feeling_loss += float(feeling_loss.detach()) * batch_size

        tp, fp, fn = get_multilabel_metrics(
            logits=genre_logits.detach(),
            targets=genre_targets,
            threshold=GENRE_THRESHOLD,
        )

        genre_tp += tp
        genre_fp += fp
        genre_fn += fn

        tp, fp, fn = get_multilabel_metrics(
            logits=feeling_logits.detach(),
            targets=feeling_targets,
            threshold=FEELING_THRESHOLD,
        )

        feeling_tp += tp
        feeling_fp += fp
        feeling_fn += fn

    return {
        "loss": total_loss / total_songs,
        "genre_loss": total_genre_loss / total_songs,
        "feeling_loss": total_feeling_loss / total_songs,
        "genre_micro_f1": micro_f1(
            genre_tp,
            genre_fp,
            genre_fn,
        ),
        "feeling_micro_f1": micro_f1(
            feeling_tp,
            feeling_fp,
            feeling_fn,
        ),
    }


def save_checkpoint(
    model: MusicAnalyzerModel0,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_loss: float,
    epochs_without_improvement: int,
    train_loss_list: list[float],
    validation_loss_list: list[float],
    train_genre_micro_f1_list: list[float],
    train_feeling_micro_f1_list: list[float],
    validation_genre_micro_f1_list: list[float],
    validation_feeling_micro_f1_list: list[float],
    path: str | Path,
    scalar_mean: FloatArray | None = None,
    scalar_std: FloatArray | None = None,
) -> None:
    """
    Saves a training checkpoint as a python dict object inside a .pt file
    with the following information:
    - epoch : `int`
        the last epoch
    - model_state_dict : `dict[str, Any]`
    - optimizer_state_dict : `dict[str, Any]`
    - best_loss : `float`
    - epochs_without_improvement : `int`
    - train_loss_list : `list[float]`
    - validation_loss_list: `list[float]`
    - train_genre_micro_f1_list : `list[float]`
    - train_feeling_micro_f1_list : `list[float]`
    - validation_genre_micro_f1_list : `list[float]`
    - validation_feeling_micro_f1_list : `list[float]`
    """
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "train_loss_list": train_loss_list,
        "validation_loss_list": validation_loss_list,
        "train_genre_micro_f1_list": train_genre_micro_f1_list,
        "train_feeling_micro_f1_list": train_feeling_micro_f1_list,
        "validation_genre_micro_f1_list": validation_genre_micro_f1_list,
        "validation_feeling_micro_f1_list": validation_feeling_micro_f1_list,
        "scalar_mean": scalar_mean,
        "scalar_std": scalar_std,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at: {path}")


def train(
    model: MusicAnalyzerModel0,
    train_data_loader: MusicDataLoader,
    validation_data_loader: MusicDataLoader,
    device: torch.device,
    name: str = "music_analyzer_model_best.pt",
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    checkpoint_path: str | Path | None = None,
) -> None:
    """
    Trains the given model with the given dataloader.
    Saves checkpoints.
    Provides rich information about the training improvement.

    Args
    --------
    model : :class:`MusicAnalyzerModel0`
    train_data_loader : :class:`MusicDataLoader`
    validation_data_loader : :class:`MusicDataLoader`
    name : `str` = "music_analyzer_model_best.pt"
        used for creating checkpoint path
    learning_rate : `float` = :data:`LEARNING_RATE`
    weight_decay : `float` = :data:`WEIGHT_DECAY`
    checkpoint_path : `str` | `Path` | `None` = `None`
        use only when loading a half trained model

    IMPORTANT
    ----------
    Use `checkpoint_path` only when you want to resume
    a training session
    """
    set_random_seed()

    genre_loss_function = nn.BCEWithLogitsLoss()
    feeling_loss_function = nn.BCEWithLogitsLoss()

    model = model.to(device)

    # The scalar normalizer is fitted on the train split
    # (`MusicDataset.fit_scalar_normalizer`) and stored in every checkpoint so
    # that inference (e.g. `demo.py`) can reproduce the same transformation.
    train_dataset = train_data_loader.dataset
    scalar_mean = getattr(train_dataset, "scalar_mean", None)
    scalar_std = getattr(train_dataset, "scalar_std", None)

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    start_epoch = 1
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    train_loss_list: list[float] = []
    validation_loss_list: list[float] = []
    train_genre_micro_f1_list: list[float] = []
    train_feeling_micro_f1_list: list[float] = []
    validation_genre_micro_f1_list: list[float] = []
    validation_feeling_micro_f1_list: list[float] = []

    if checkpoint_path is not None:
        saved_data = torch.load(
            Path(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(saved_data["model_state_dict"])
        optimizer.load_state_dict(saved_data["optimizer_state_dict"])
        start_epoch = saved_data["epoch"] + 1
        best_validation_loss = saved_data["best_loss"]
        epochs_without_improvement = saved_data["epochs_without_improvement"]
        train_loss_list = saved_data["train_loss_list"]
        validation_loss_list = saved_data["validation_loss_list"]
        train_genre_micro_f1_list = saved_data["train_genre_micro_f1_list"]
        train_feeling_micro_f1_list = saved_data["train_feeling_micro_f1_list"]
        validation_genre_micro_f1_list = saved_data["validation_genre_micro_f1_list"]
        validation_feeling_micro_f1_list = saved_data[
            "validation_feeling_micro_f1_list"
        ]

    checkpoint_path: Path = CHECKPOINT_DIR / Path(name).with_suffix(".pt")
    save_path: Path = SAVED_MODELS_DIR / Path(name).with_suffix(".pt")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_data_loader,
            device=device,
            genre_loss_function=genre_loss_function,
            feeling_loss_function=feeling_loss_function,
            optimizer=optimizer,
        )

        validation_metrics = run_epoch(
            model=model,
            data_loader=validation_data_loader,
            device=device,
            genre_loss_function=genre_loss_function,
            feeling_loss_function=feeling_loss_function,
            optimizer=None,
        )

        scheduler.step(validation_metrics["loss"])

        current_learning_rate = optimizer.param_groups[0]["lr"]
        train_loss_list.append(train_metrics["loss"])
        validation_loss_list.append(validation_metrics["loss"])
        train_genre_micro_f1_list.append(train_metrics["genre_micro_f1"])
        train_feeling_micro_f1_list.append(train_metrics["feeling_micro_f1"])
        validation_genre_micro_f1_list.append(validation_metrics["genre_micro_f1"])
        validation_feeling_micro_f1_list.append(validation_metrics["feeling_micro_f1"])

        # Metrics
        print(
            f"Epoch {epoch:02d}/{EPOCHS}\n"
            f"LR: {current_learning_rate:.2e}\n"
            "Train:"
            f"\tloss: {train_metrics['loss']:.4f}\n"
            f"\tgenre_f1: {train_metrics['genre_micro_f1']:.4f}\n"
            f"\tfeeling_f1: {train_metrics['feeling_micro_f1']:.4f}\n"
            "Validation:"
            f"\tloss: {validation_metrics['loss']:.4f}\n"
            f"\tgenre_f1: {validation_metrics['genre_micro_f1']:.4f}\n"
            f"\tfeeling_f1: {validation_metrics['feeling_micro_f1']:.4f}\n"
            "\n"
        )

        if validation_metrics["loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
                train_loss_list=train_loss_list,
                validation_loss_list=validation_loss_list,
                train_genre_micro_f1_list=train_genre_micro_f1_list,
                train_feeling_micro_f1_list=train_feeling_micro_f1_list,
                validation_genre_micro_f1_list=validation_genre_micro_f1_list,
                validation_feeling_micro_f1_list=validation_feeling_micro_f1_list,
                scalar_mean=scalar_mean,
                scalar_std=scalar_std,
                path=checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping: validation loss did not improve"
                f"for {EARLY_STOPPING_PATIENCE} epochs."
            )
            break

    if checkpoint_path.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
        copy(src=checkpoint_path, dst=save_path)
        print(f"Training complete! Best validation loss: {best_validation_loss:.4f}")
        print(f"Best model: {save_path}")
    else:
        print("Training was interrupted before the first checkpoint was saved.")

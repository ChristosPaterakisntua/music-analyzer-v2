import torch
from torch import nn

from .dataset import MusicDataLoader
from .models import MusicAnalyzerModel0
from .train import run_epoch


def test(
    model: MusicAnalyzerModel0,
    data_loader: MusicDataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Get test metrics from a trained model

    Args
    -------
    model : :class:`MusicAnalyzerModel0`
        trained model
    dataloader : :class:`MusicDataLoader`
        loads the test dataset
    device : `torch.device`

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
    return run_epoch(
        model=model,
        data_loader=data_loader,
        device=device,
        genre_loss_function=nn.BCEWithLogitsLoss(),
        feeling_loss_function=nn.BCEWithLogitsLoss(),
        optimizer=None,
    )

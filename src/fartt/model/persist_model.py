from pathlib import Path

import torch
from torch import nn


def save_model(model: nn.Module, path: Path) -> None:
    """
    Save a trained model to disk. Saves the whole module (architecture and
    weights together), not just a state_dict -- unlike the retired N-BEATS
    path, this MLP has no separate config object to reconstruct the
    architecture from at load time.

    Parameters
    ----------
    - model (nn.Module): Trained model to save.
    - path (Path): File path to save the model to.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)


def load_model(path: Path) -> nn.Module:
    """
    Load a model previously saved with `save_model`, in eval mode.

    Parameters
    ----------
    - path (Path): File path to load the model from.

    Returns
    -------
    - nn.Module: The loaded model.

    """
    # weights_only=False is required to unpickle a full nn.Module (rather
    # than just tensors/state_dict) -- safe here since these are
    # self-produced local artifacts, not untrusted downloads.
    model = torch.load(path, weights_only=False)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] -- torch.load's overloads are untyped upstream (torch/serialization.py)
    model.eval()
    return model

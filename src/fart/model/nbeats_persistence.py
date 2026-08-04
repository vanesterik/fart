from pathlib import Path
from typing import Any, Dict, cast

import torch

from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig


def save_model(model: NBeatsNet, config: NBeatsConfig, path: Path) -> None:
    """
    Save a trained NBeatsNet's weights and config as a single,
    self-describing checkpoint file.

    Parameters
    ----------
    - model (NBeatsNet): Trained model to save.
    - config (NBeatsConfig): Config the model was built and trained with.
    - path (Path): Destination file path. Parent directories are created if
      they don't exist.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: Dict[str, Any] = {
        "state_dict": model.state_dict(),
        "config": config.model_dump(),
    }
    torch.save(checkpoint, path)


def load_model(path: Path) -> NBeatsNet:
    """
    Load a saved NBeatsNet checkpoint, reconstructing the architecture from
    its bundled config before loading weights into it. Always loads onto
    CPU -- callers move the model to a specific device themselves if
    needed.

    Parameters
    ----------
    - path (Path): Path to a checkpoint previously saved by `save_model`.

    Returns
    -------
    - NBeatsNet: The reconstructed model, weights loaded, in eval mode.

    """
    checkpoint = cast(
        Dict[str, Any], torch.load(path, map_location="cpu", weights_only=True)
    )
    config = NBeatsConfig(**checkpoint["config"])
    model = NBeatsNet(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

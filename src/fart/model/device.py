import torch


def get_device() -> torch.device:
    """
    Auto-detect the best available torch device for this machine.

    Checks Apple Silicon MPS only, not CUDA -- `torch` is pinned to the
    CPU-only wheel index (`pyproject.toml`) on Linux/Windows, so
    `torch.cuda.is_available()` can never be True in this project.

    Returns
    -------
    - torch.device: `mps` if available, otherwise `cpu`.

    """
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

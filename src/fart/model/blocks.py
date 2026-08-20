from torch import nn


def linear_block(in_features: int, out_features: int, dropout: float) -> nn.Sequential:
    """
    Build one reusable feed-forward "neural block":
    Linear -> BatchNorm1d -> ReLU -> Dropout.

    Parameters
    ----------
    - in_features (int): Size of the block's input.
    - out_features (int): Size of the block's output (and of the
      BatchNorm1d/ReLU/Dropout that follow).
    - dropout (float): Dropout probability, in [0, 1).

    Returns
    -------
    - nn.Sequential: The four-layer block, ready to be composed into a
      larger `nn.Sequential` by a builder.

    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )


def conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    dropout: float,
) -> nn.Sequential:
    """
    Build one reusable convolutional "neural block":
    Conv1d -> BatchNorm1d -> ReLU -> Dropout.

    Uses `padding="same"` so the block never shrinks its input's sequence
    length, regardless of `kernel_size` or how many blocks are stacked --
    keeps builders that pool globally afterward free of manual output-length
    arithmetic.

    Parameters
    ----------
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels (and of the
      BatchNorm1d that follows).
    - kernel_size (int): Convolution kernel width.
    - dropout (float): Dropout probability, in [0, 1).

    Returns
    -------
    - nn.Sequential: The four-layer block, ready to be composed into a
      larger `nn.Sequential` by a builder.

    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same"),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(p=dropout),
    )

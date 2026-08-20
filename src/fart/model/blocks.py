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

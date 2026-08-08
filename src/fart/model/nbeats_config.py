from pydantic import BaseModel


class NBeatsConfig(BaseModel):
    """
    Configuration for the quick-prototype N-BEATS model.

    Attributes
    ----------
    - batch_size (int): Minibatch size for training and inference.
    - beta_nll (float): Exponent weighting each window's loss contribution
      by its own predicted variance (beta-NLL, Seitzer et al. 2022). `0.0`
      recovers plain Gaussian NLL; `1.0` approximates MSE-style weighting
      on the mean.
    - epochs (int): Number of training epochs.
    - hidden_width (int): Width of each block's hidden fully-connected layers.
    - learning_rate (float): Adam optimizer learning rate.
    - lookback (int): Backcast length, in candles.
    - num_blocks_per_stack (int): Number of blocks per stack.
    - num_stacks (int): Number of stacks of blocks.

    """

    batch_size: int = 128
    beta_nll: float = 0.5
    epochs: int = 50
    hidden_width: int = 64
    learning_rate: float = 1e-3
    lookback: int = 30
    num_blocks_per_stack: int = 3
    num_stacks: int = 2

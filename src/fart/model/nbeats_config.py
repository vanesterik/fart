from pydantic import BaseModel


class NBeatsConfig(BaseModel):
    """
    Configuration for the quick-prototype N-BEATS model.

    Attributes
    ----------
    - batch_size (int): Minibatch size for training and inference.
    - beta_nll (float): Exponent weighting each window's loss contribution
      by its own predicted variance (beta-NLL, Seitzer et al. 2022). `0.0`
      recovers plain Gaussian NLL exactly (see
      `test_nbeats_loss.py::test_beta_nll_loss_at_beta_zero_matches_gaussian_nll_loss`);
      `1.0` approximates MSE-style weighting on the mean. Defaults to `0.0`
      -- a 130-run reproducibility check (30 runs each at `0.0`/`0.5`/`1.0`,
      `notebooks/3.0-kve-beta-nll-reproducibility-check.ipynb`) found no
      statistically significant difference between any of the three in
      either mean confidence/error correlation or run-to-run variance, so
      beta-NLL is not proven to fix the underlying variance-collapse problem
      (confidence stays weakly informative, r ~= -0.09, at every value
      tested). Kept as a config knob rather than reverting to a hardcoded
      `nn.GaussianNLLLoss()` in case it's worth revisiting with a larger
      training set.
    - epochs (int): Number of training epochs.
    - hidden_width (int): Width of each block's hidden fully-connected layers.
    - learning_rate (float): Adam optimizer learning rate.
    - lookback (int): Backcast length, in candles.
    - num_blocks_per_stack (int): Number of blocks per stack.
    - num_stacks (int): Number of stacks of blocks.

    """

    batch_size: int = 128
    beta_nll: float = 0.0
    epochs: int = 50
    hidden_width: int = 64
    learning_rate: float = 1e-3
    lookback: int = 30
    num_blocks_per_stack: int = 3
    num_stacks: int = 2

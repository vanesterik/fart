from typing import Tuple

import numpy as np
import polars as pl
import torch


def build_return_windows(
    close_prices: pl.Series,
    lookback: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Turns a chronological Close-price series into sliding windows of
    percent returns, paired with the next return as the training target.

    Parameters
    ----------
    - close_prices (pl.Series): Chronological Close prices.
    - lookback (int): Number of past returns per input window.

    Returns
    -------
    - Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
        - X (torch.Tensor): Shape (num_windows, lookback), float32.
        - y (torch.Tensor): Shape (num_windows,), float32 — the next-step
          return immediately following each window.

    """
    prices = close_prices.to_numpy()
    returns = (prices[1:] - prices[:-1]) / prices[:-1]

    num_windows = len(returns) - lookback
    if num_windows <= 0:
        raise ValueError(
            f"Not enough data to build a single window: need at least "
            f"{lookback + 2} close prices for lookback={lookback}, got {len(prices)}."
        )

    X = np.stack([returns[i : i + lookback] for i in range(num_windows)])
    y = returns[lookback : lookback + num_windows]

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )

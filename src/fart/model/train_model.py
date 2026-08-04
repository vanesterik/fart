from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import polars as pl
import torch
from loguru import logger
from torch import nn

from fart.constants import TIMESTAMP
from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig
from fart.model.nbeats_dataset import build_return_windows
from fart.model.train_test_split import train_test_split
from fart.utils import get_candle_filepath


def prepare_training_data(
    data_dir: Path,
    market: str,
    interval: str,
    months: Optional[int] = 6,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    filepath = get_candle_filepath(data_dir, market, interval)

    if not filepath.exists():
        raise FileNotFoundError(
            f"No candle data found at '{filepath}'. Run 'fart download' first."
        )

    df = pl.read_csv(filepath)

    if months is not None:
        max_timestamp = df[TIMESTAMP].max()
        assert isinstance(max_timestamp, int)
        cutoff = max_timestamp - months * 30 * 24 * 60 * 60 * 1000
        df = df.filter(pl.col(TIMESTAMP) >= cutoff)

    df = calculate_technical_indicators(df)
    df = df.fill_nan(None).drop_nulls()

    return train_test_split(df)


def train(
    data_dir: Path,
    market: str,
    interval: str,
    months: Optional[int] = 6,
    config: Optional[NBeatsConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    config = config or NBeatsConfig()

    X_train, X_test, y_train, y_test = prepare_training_data(
        data_dir, market, interval, months
    )
    logger.info(
        f"Prepared training data: X_train={X_train.shape}, "
        f"X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}"
    )

    n_train = y_train.shape[0]
    close_prices = pl.concat([y_train, y_test])
    X_all, y_all = build_return_windows(close_prices, config.lookback)

    n_train_windows = max(0, n_train - config.lookback - 1)
    X_train_windows, y_train_windows = X_all[:n_train_windows], y_all[:n_train_windows]
    X_test_windows = X_all[n_train_windows:]

    model = NBeatsNet(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.GaussianNLLLoss()

    model.train()
    for _ in range(config.epochs):
        optimizer.zero_grad()
        output = model(X_train_windows)
        mu, log_sigma = output.unbind(-1)
        loss = loss_fn(mu, y_train_windows, log_sigma.exp() ** 2)
        loss.backward()
        optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    model.eval()
    with torch.no_grad():
        output = model(X_test_windows)
        mu, log_sigma = output.unbind(-1)
        magnitudes = mu.numpy()
        confidences = (1 / (1 + log_sigma.exp())).numpy()

    logger.info(
        f"N-BEATS quick prototype: {len(magnitudes)} test candles, "
        f"magnitude mean={magnitudes.mean():.5f} std={magnitudes.std():.5f}, "
        f"confidence mean={confidences.mean():.5f}"
    )

    return magnitudes, confidences

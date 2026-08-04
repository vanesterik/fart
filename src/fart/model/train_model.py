from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from loguru import logger
from tabulate import tabulate
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from fart.constants import TIMESTAMP
from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.features.sort_and_deduplicate_candles import sort_and_deduplicate_candles
from fart.model.device import get_device
from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig
from fart.model.nbeats_dataset import build_return_windows
from fart.model.nbeats_persistence import save_model
from fart.model.train_test_split import train_test_split
from fart.utils import get_candle_filepath, get_model_filepath


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
    df = sort_and_deduplicate_candles(df)

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
    artifacts_dir: Path,
    months: Optional[int] = 6,
    config: Optional[NBeatsConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    config = config or NBeatsConfig()
    device = device or get_device()

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

    model = NBeatsNet(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.GaussianNLLLoss()

    train_loader = DataLoader(
        TensorDataset(X_train_windows, y_train_windows),
        batch_size=config.batch_size,
        shuffle=True,
    )

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            mu, log_sigma = output.unbind(-1)
            loss = loss_fn(mu, y_batch, log_sigma.exp() ** 2)
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType] -- Adam.step is untyped upstream (torch/optim/adam.py)
            epoch_loss += loss.item() * X_batch.shape[0]

        logger.info(
            f"Epoch {epoch + 1}/{config.epochs}: "
            f"loss={epoch_loss / len(X_train_windows):.6f}"
        )

    test_loader = DataLoader(
        TensorDataset(X_test_windows), batch_size=config.batch_size, shuffle=False
    )

    model.eval()
    mu_batches: List[Tensor] = []
    log_sigma_batches: List[Tensor] = []
    with torch.no_grad():
        for (X_batch,) in test_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            mu, log_sigma = output.unbind(-1)
            mu_batches.append(mu.cpu())
            log_sigma_batches.append(log_sigma.cpu())

    magnitudes = torch.cat(mu_batches).numpy()
    confidences = (1 / (1 + torch.cat(log_sigma_batches).exp())).numpy()

    results = {
        "test candles": len(magnitudes),
        "device": str(device),
        "magnitude mean": f"{magnitudes.mean():.5f}",
        "magnitude std": f"{magnitudes.std():.5f}",
        "confidence mean": f"{confidences.mean():.5f}",
    }
    table = tabulate(results.items())
    logger.info(f"\n\n{table}\n")

    timestamp = datetime.now(timezone.utc)
    model_path = get_model_filepath(artifacts_dir, market, interval, timestamp)
    save_model(model.cpu(), config, model_path)
    logger.info(f"Saved model artifact to {model_path}")

    return magnitudes, confidences

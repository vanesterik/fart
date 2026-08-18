from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType] -- sklearn ships no type stubs (sklearn/metrics/__init__.py)
from sklearn.model_selection import TimeSeriesSplit  # pyright: ignore[reportMissingTypeStubs] -- sklearn ships no type stubs (sklearn/model_selection/__init__.py)
from tabulate import tabulate
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fart.model.evaluate_model import calculate_accuracy


def build_model(num_lags: int, hidden_width: int) -> nn.Module:
    """
    Build a small feed-forward regression model: two ReLU hidden layers
    over a window of `num_lags` past values, predicting the next value.

    Parameters
    ----------
    - num_lags (int): Width of the input window (matches the lag window
      size produced by `prepare_datasets.py::prepare_datasets`).
    - hidden_width (int): Width of each hidden layer.

    Returns
    -------
    - nn.Module: An untrained model.

    """
    return nn.Sequential(
        nn.Linear(num_lags, hidden_width),
        nn.ReLU(),
        nn.Linear(hidden_width, hidden_width),
        nn.ReLU(),
        nn.Linear(hidden_width, 1),
    )


def train_model(
    build_model_fn: Callable[[], nn.Module],
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    n_splits: int = 5,
    max_train_size: int | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """
    Fit a model on `x_train`/`y_train`, using time-series cross-validation
    to get a robustness signal before committing to a single training run.

    A fresh model is built (via `build_model_fn`) and trained for each of
    `n_splits` expanding-window folds (`sklearn.model_selection.TimeSeriesSplit`
    -- chronological, not random/shuffled, since shuffled k-fold would leak
    future data into training on time series). Reusing one model across
    folds would leak weights/optimizer state forward across time, which is
    why a fresh model is required per fold. After folds, a final model is
    trained on the complete `x_train`/`y_train` and returned -- CV is a
    diagnostic on top of training, not a replacement for it.

    Parameters
    ----------
    - build_model_fn (Callable[[], nn.Module]): Zero-arg factory returning
      an untrained model, called once per fold plus once for the final
      training pass.
    - x_train (np.ndarray): Training windows, shape (n, num_lags).
    - y_train (np.ndarray): Training targets, shape (n,).
    - batch_size (int): Minibatch size.
    - learning_rate (float): Adam optimizer learning rate.
    - num_epochs (int): Number of training epochs per fold and for the
      final pass.
    - n_splits (int): Number of CV folds. `1` disables CV entirely (only
      the final pass runs, `cv_results` is empty) -- useful for fast
      tests/iteration even though the default runs CV.
    - max_train_size (Optional[int]): Caps each fold's training window to
      a fixed size (rolling window) instead of the default expanding
      window. Expanding folds are bigger later, which confounds "more
      data" with "more recent data"; a fixed size isolates recency.

    Returns
    -------
    - Tuple[nn.Module, list[dict[str, float]]]: The model trained on all of
      `x_train`/`y_train`, and `cv_results` -- one record per (fold, epoch)
      with `fold`, `epoch`, `train_loss`, `val_rmse`, `val_mae`,
      `val_accuracy` (empty list if `n_splits == 1`), suitable for plotting
      a train-vs-validation learning curve per fold.

    """
    cv_results: list[dict[str, float]] = []
    num_fit_passes = (n_splits if n_splits > 1 else 0) + 1
    progress_bar = tqdm(total=num_epochs * num_fit_passes)

    if n_splits > 1:
        splitter = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size)
        fold_summaries: list[list[object]] = []
        folds = splitter.split(x_train)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType] -- TimeSeriesSplit.split is untyped upstream (sklearn ships no type stubs)
        for fold, (train_idx, val_idx) in enumerate(folds, start=1):
            _, history = _fit(
                model=build_model_fn(),
                x_train=x_train[train_idx],
                y_train=y_train[train_idx],
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                progress_bar=progress_bar,
                x_val=x_train[val_idx],
                y_val=y_train[val_idx],
            )
            for record in history:
                cv_results.append({**record, "fold": float(fold)})
            last = history[-1]
            fold_summaries.append(
                [
                    fold,
                    len(train_idx),
                    len(val_idx),
                    round(last["val_rmse"], 6),
                    round(last["val_mae"], 6),
                    round(last["val_accuracy"], 2),
                ]
            )
        table = tabulate(
            fold_summaries,
            headers=["Fold", "n_train", "n_val", "val_rmse", "val_mae", "val_accuracy"],
        )
        logger.info(f"\n\n{table}\n")

    model, _ = _fit(
        model=build_model_fn(),
        x_train=x_train,
        y_train=y_train,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        progress_bar=progress_bar,
    )
    progress_bar.close()

    return model, cv_results


def _fit(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    progress_bar: tqdm[Any],
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """
    Fit `model` on `x_train`/`y_train`, advancing the shared `progress_bar`
    by one step per epoch. If `x_val`/`y_val` are given, each epoch's
    history record also carries validation RMSE/MAE/accuracy (`history` is
    `[]` when they're omitted, since nothing consumes an unvalidated fit's
    per-epoch history).

    """
    train_dataloader = init_dataloader(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
    )
    loss_fn = nn.MSELoss()
    optimizer = init_optimizer(
        model=model,
        learning_rate=learning_rate,
    )

    x_val_tensor = (
        torch.tensor(x_val, dtype=torch.float32) if x_val is not None else None
    )

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            output = model(x_batch).squeeze(-1)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType] -- Adam.step is untyped upstream (torch/optim/adam.py)
            epoch_loss += loss.item() * x_batch.shape[0]
        train_loss = epoch_loss / len(x_train)

        if x_val_tensor is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val_tensor).squeeze(-1).numpy()
            model.train()
            val_rmse = float(root_mean_squared_error(y_val, val_pred))
            val_mae = float(mean_absolute_error(y_val, val_pred))  # pyright: ignore[reportUnknownArgumentType] -- mean_absolute_error's return is untyped upstream (sklearn ships no type stubs)
            val_accuracy = calculate_accuracy(
                val_pred.reshape(-1, 1), y_val.reshape(-1, 1)
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "val_rmse": val_rmse,
                    "val_mae": val_mae,
                    "val_accuracy": val_accuracy,
                }
            )

        progress_bar.update(1)  # pyright: ignore[reportUnknownMemberType] -- tqdm.update is untyped upstream (tqdm/std.py)

    return model, history


def init_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return cast(
        DataLoader[tuple[torch.Tensor, torch.Tensor]],
        DataLoader(
            TensorDataset(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=True,
        ),
    )


def init_optimizer(
    model: nn.Module,
    learning_rate: float,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

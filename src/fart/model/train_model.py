from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> tuple[nn.Module, list[dict[str, float]]]:
    """
    Fit `model` on `x_train`/`y_train` for `num_epochs`, in place.

    If `x_val`/`y_val` are given, each epoch's loss is also evaluated
    against them and recorded in `history` alongside `train_loss`, to spot
    overfitting (a validation loss that diverges from the training loss).
    `history` is `[]` when they're omitted, since nothing consumes an
    unvalidated fit's per-epoch history.

    Parameters
    ----------
    - model (nn.Module): Untrained model to fit, updated in place.
    - x_train (np.ndarray): Training windows, shape (n_train, num_lags).
    - y_train (np.ndarray): Training targets, shape (n_train,).
    - batch_size (int): Minibatch size.
    - learning_rate (float): Adam optimizer learning rate.
    - num_epochs (int): Number of training epochs.
    - x_val (Optional[np.ndarray]): Held-out validation windows, same
      shape convention as `x_train`. Pass alongside `y_val` to get a
      per-epoch train-vs-validation loss history.
    - y_val (Optional[np.ndarray]): Held-out validation targets, paired
      with `x_val`.

    Returns
    -------
    - Tuple[nn.Module, list[dict[str, float]]]: The trained model, and
      `history` -- one record per epoch with `epoch`, `train_loss`, and
      (if `x_val`/`y_val` were given) `val_loss`, suitable for plotting a
      train-vs-validation learning curve.

    """
    train_dataloader = init_dataloader(x=x_train, y=y_train, batch_size=batch_size)
    loss_fn = nn.MSELoss()
    optimizer = init_optimizer(model=model, learning_rate=learning_rate)

    history: list[dict[str, float]] = []
    for epoch in tqdm(range(num_epochs), desc="Training"):  # pyright: ignore[reportUnknownMemberType] -- tqdm's __init__ overloads are untyped upstream (tqdm/std.py)
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )

        if x_val is not None and y_val is not None:
            val_loss = _validate(
                model=model,
                x_val=x_val,
                y_val=y_val,
                loss_fn=loss_fn,
            )
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )

    return model, history


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    """
    Run one training epoch over `dataloader`, updating `model`'s weights
    in place.

    Parameters
    ----------
    - model (nn.Module): Model to train, updated in place.
    - dataloader (DataLoader[tuple[torch.Tensor, torch.Tensor]]): Minibatches
      of `(x_batch, y_batch)`.
    - optimizer (torch.optim.Optimizer): Optimizer stepping `model`'s
      parameters.
    - loss_fn (nn.Module): Loss criterion.

    Returns
    -------
    - float: The epoch's mean per-sample loss.

    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        output = model(x_batch).squeeze(-1)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()  # pyright: ignore[reportUnknownMemberType] -- Adam.step is untyped upstream (torch/optim/adam.py)
        total_loss += loss.item() * x_batch.shape[0]
        total_samples += x_batch.shape[0]

    return total_loss / total_samples


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator] -- torch.no_grad's decorator overload is untyped upstream (torch/autograd/grad_mode.py)
def _validate(
    model: nn.Module,
    x_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn: nn.Module,
) -> float:
    """
    Evaluate `model` against a held-out validation set, without updating
    weights.

    Parameters
    ----------
    - model (nn.Module): Model to evaluate.
    - x_val (np.ndarray): Validation windows.
    - y_val (np.ndarray): Validation targets, paired with `x_val`.
    - loss_fn (nn.Module): Loss criterion, same one used for training so
      `train_loss`/`val_loss` stay directly comparable.

    Returns
    -------
    - float: The validation loss.

    """
    model.eval()
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    output = model(x_val_tensor).squeeze(-1)

    return loss_fn(output, y_val_tensor).item()


def init_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """
    Wrap `x`/`y` in a shuffled `DataLoader` of `(x_batch, y_batch)` tensor
    pairs, for minibatch training.

    Shuffling here is minibatch order, not the time series itself: `x`/`y`
    are already fixed, chronologically-built lag windows confined to one
    split (see `prepare_datasets.py::train_test_split`), so reordering
    which examples land in which minibatch doesn't leak future data --
    it just avoids the optimizer (and `BatchNorm1d`) seeing the same
    temporally-correlated run of windows every epoch.

    Parameters
    ----------
    - x (np.ndarray): Input windows.
    - y (np.ndarray): Targets, paired with `x`.
    - batch_size (int): Minibatch size.

    Returns
    -------
    - DataLoader[tuple[torch.Tensor, torch.Tensor]]: Yields shuffled
      `(x_batch, y_batch)` tensor pairs each epoch.

    """
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
    """
    Build an Adam optimizer over `model`'s parameters.

    Parameters
    ----------
    - model (nn.Module): Model whose parameters the optimizer will step.
    - learning_rate (float): Adam learning rate.

    Returns
    -------
    - torch.optim.Optimizer: An Adam optimizer bound to `model`'s
      parameters.

    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

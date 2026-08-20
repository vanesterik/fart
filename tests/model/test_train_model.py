import copy

import numpy as np
import torch
from torch import nn

from fart.model.train_model import init_dataloader, init_optimizer, train_model


def test_init_dataloader_batches_and_converts_to_tensors() -> None:
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10, dtype=np.float32)

    loader = init_dataloader(x=x, y=y, batch_size=4)
    batches = list(loader)

    assert sum(x_batch.shape[0] for x_batch, _ in batches) == 10
    x_batch, y_batch = batches[0]
    assert isinstance(x_batch, torch.Tensor)
    assert x_batch.dtype == torch.float32
    assert y_batch.dtype == torch.float32


def test_init_optimizer_returns_adam_with_learning_rate() -> None:
    model = nn.Linear(2, 1)

    optimizer = init_optimizer(model=model, learning_rate=0.01)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_train_model_reduces_loss() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 3)).astype(np.float32)
    true_weights = np.array([1.5, -2.0, 0.5], dtype=np.float32)
    y = (x @ true_weights).astype(np.float32)

    loss_fn = nn.MSELoss()

    def mse(m: nn.Module) -> float:
        with torch.no_grad():
            pred = m(torch.tensor(x)).squeeze(-1)
            return loss_fn(pred, torch.tensor(y)).item()

    torch.manual_seed(0)
    model = nn.Linear(3, 1)
    # train_model fits `model` in place, so snapshot its initial weights
    # here to measure "before" loss independently of the trained model.
    loss_before = mse(copy.deepcopy(model))

    trained, history = train_model(
        model=model,
        x_train=x,
        y_train=y,
        batch_size=16,
        learning_rate=0.05,
        num_epochs=50,
    )

    loss_after = mse(trained)

    assert loss_after < loss_before
    assert history == []


def test_train_model_with_validation_returns_per_epoch_history() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2)).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)
    x_val = rng.normal(size=(10, 2)).astype(np.float32)
    y_val = rng.normal(size=10).astype(np.float32)
    num_epochs = 3

    _, history = train_model(
        model=nn.Linear(2, 1),
        x_train=x,
        y_train=y,
        x_val=x_val,
        y_val=y_val,
        batch_size=8,
        learning_rate=0.01,
        num_epochs=num_epochs,
    )

    assert len(history) == num_epochs
    assert [record["epoch"] for record in history] == [1.0, 2.0, 3.0]
    for record in history:
        assert set(record.keys()) == {"epoch", "train_loss", "val_loss"}

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

    def build_model_fn() -> nn.Module:
        # Reseeded on every call so the "before" model measured here and
        # the model train_model() builds internally start from identical
        # weights -- build_model_fn is a factory, not a shared instance.
        torch.manual_seed(0)
        return nn.Linear(3, 1)

    def mse(m: nn.Module) -> float:
        with torch.no_grad():
            pred = m(torch.tensor(x)).squeeze(-1)
            return loss_fn(pred, torch.tensor(y)).item()

    loss_before = mse(build_model_fn())

    trained, cv_results = train_model(
        build_model_fn=build_model_fn,
        x_train=x,
        y_train=y,
        batch_size=16,
        learning_rate=0.05,
        num_epochs=50,
        num_splits=1,
    )

    loss_after = mse(trained)

    assert loss_after < loss_before
    assert cv_results == []


def test_train_model_cv_returns_per_fold_per_epoch_history() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2)).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)
    num_epochs = 3
    num_splits = 3

    _, cv_results = train_model(
        build_model_fn=lambda: nn.Linear(2, 1),
        x_train=x,
        y_train=y,
        batch_size=8,
        learning_rate=0.01,
        num_epochs=num_epochs,
        num_splits=num_splits,
    )

    assert len(cv_results) == num_splits * num_epochs
    folds = {record["fold"] for record in cv_results}
    assert folds == {1.0, 2.0, 3.0}
    for fold in folds:
        epochs = sorted(
            record["epoch"] for record in cv_results if record["fold"] == fold
        )
        assert epochs == [1.0, 2.0, 3.0]
    for record in cv_results:
        assert set(record.keys()) == {"fold", "epoch", "train_loss", "val_loss"}


def test_train_model_cv_builds_a_fresh_model_per_fold() -> None:
    call_count = 0

    def build_model_fn() -> nn.Module:
        nonlocal call_count
        call_count += 1
        return nn.Linear(2, 1)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2)).astype(np.float32)
    y = rng.normal(size=40).astype(np.float32)

    train_model(
        build_model_fn=build_model_fn,
        x_train=x,
        y_train=y,
        batch_size=8,
        learning_rate=0.01,
        num_epochs=2,
        num_splits=3,
    )

    # 3 folds + 1 final pass on the complete training set.
    assert call_count == 4

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, root_mean_squared_error  # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType] -- sklearn ships no type stubs (sklearn/metrics/__init__.py)
from torch import nn


def evaluate_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    model.eval()
    with torch.no_grad():
        y_train_pred = (
            model(torch.tensor(x_train, dtype=torch.float32)).numpy().reshape(-1)
        )
        y_test_pred = (
            model(torch.tensor(x_test, dtype=torch.float32)).numpy().reshape(-1)
        )

    accuracy_train = calculate_accuracy(y_train_pred, y_train)
    accuracy_test = calculate_accuracy(y_test_pred, y_test)
    rmse_train = float(root_mean_squared_error(y_train_pred, y_train))
    rmse_test = float(root_mean_squared_error(y_test_pred, y_test))
    mae_train = float(mean_absolute_error(y_train_pred, y_train))  # pyright: ignore[reportUnknownArgumentType] -- mean_absolute_error's return is untyped upstream (sklearn ships no type stubs)
    mae_test = float(mean_absolute_error(y_test_pred, y_test))  # pyright: ignore[reportUnknownArgumentType] -- mean_absolute_error's return is untyped upstream (sklearn ships no type stubs)

    return (
        y_train_pred,
        y_test_pred,
        accuracy_train,
        accuracy_test,
        rmse_train,
        rmse_test,
        mae_train,
        mae_test,
    )


def calculate_accuracy(
    predicted_returns: np.ndarray,
    real_returns: np.ndarray,
) -> float:
    """
    Calculate the directional accuracy of predicted returns against
    realized returns, as the percentage of samples where predicted and
    real returns share the same sign.

    Parameters
    ----------
    - predicted_returns (np.ndarray): Predicted returns.
    - real_returns (np.ndarray): Realized returns, same length as
      `predicted_returns`.

    Returns
    -------
    - float: Percentage of samples where
      `sign(predicted_returns) == sign(real_returns)`, in `[0, 100]`.

    """
    hits = np.sum(np.sign(predicted_returns) == np.sign(real_returns))
    total_samples = len(predicted_returns)

    return float(hits / total_samples * 100)

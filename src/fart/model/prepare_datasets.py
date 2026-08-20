from pathlib import Path

import numpy as np
import polars as pl

from fart.features.calculate_magnitude import calculate_magnitude
from fart.features.sort_and_deduplicate import sort_and_deduplicate


def prepare_datasets(
    data_filepath: Path,
    target: str,
    num_lags: int,
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Prepares the data for training, validation, and testing by loading the
    data from a CSV file, sorting and deduplicating it by timestamp,
    calculating the magnitude of the target column, and splitting it into
    training, validation, and test sets.

    Parameters
    ----------
    - data_filepath (Path): Path to the CSV file containing the data.
    - target (str): The name of the target column in the DataFrame.
    - num_lags (int): Number of past values per input window.
    - train_size (float): The proportion of windows to include in the
      training split.
    - val_size (float): The proportion of windows to include in the
      validation split. The remainder (`1 - train_size - val_size`)
      becomes the test split.

    Returns
    -------
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
      A tuple containing:
        - x_train (np.ndarray): Training windows, shape (n_train, num_lags), float32.
        - y_train (np.ndarray): Training targets, shape (n_train,), float32.
        - x_val (np.ndarray): Validation windows, shape (n_val, num_lags), float32.
        - y_val (np.ndarray): Validation targets, shape (n_val,), float32.
        - x_test (np.ndarray): Test windows, shape (n_test, num_lags), float32.
        - y_test (np.ndarray): Test targets, shape (n_test,), float32.

    """
    df = pl.read_csv(data_filepath)
    df = sort_and_deduplicate(df)
    df = calculate_magnitude(df)
    df = df.fill_nan(None).drop_nulls()

    data = df[target].to_numpy().astype(np.float32)

    return train_test_split(
        data=data, num_lags=num_lags, train_size=train_size, val_size=val_size
    )


def train_test_split(
    data: np.ndarray,
    num_lags: int,
    train_size: float = 0.6,
    val_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Turns a chronological target column into sliding windows of lagged
    values, paired with the next value as the training target, then splits
    the windows into training, validation, and test sets by row order (no
    shuffling, since this is time series data).

    Parameters
    ----------
    - data (np.ndarray): A numpy array containing the target values, in
      chronological order.
    - num_lags (int): Number of past values per input window.
    - train_size (float): The proportion of windows to include in the
      training split.
    - val_size (float): The proportion of windows to include in the
      validation split. The remainder (`1 - train_size - val_size`)
      becomes the test split.

    Returns
    -------
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
      A tuple containing:
        - x_train (np.ndarray): Training windows, shape (n_train, num_lags), float32.
        - y_train (np.ndarray): Training targets, shape (n_train,), float32.
        - x_val (np.ndarray): Validation windows, shape (n_val, num_lags), float32.
        - y_val (np.ndarray): Validation targets, shape (n_val,), float32.
        - x_test (np.ndarray): Test windows, shape (n_test, num_lags), float32.
        - y_test (np.ndarray): Test targets, shape (n_test,), float32.

    """

    num_windows = len(data) - num_lags
    if num_windows <= 0:
        raise ValueError(
            f"Not enough data to build a single window: need at least "
            f"{num_lags + 1} rows for num_lags={num_lags}, got {len(data)}."
        )

    x = np.stack([data[i : i + num_lags] for i in range(num_windows)])
    y = data[num_lags : num_lags + num_windows]

    train_end = int(train_size * num_windows)
    val_end = train_end + int(val_size * num_windows)

    x_train = x[:train_end]
    y_train = y[:train_end]
    x_val = x[train_end:val_end]
    y_val = y[train_end:val_end]
    x_test = x[val_end:]
    y_test = y[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test

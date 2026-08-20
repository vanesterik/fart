from pathlib import Path

import numpy as np
import pytest

from fart.constants import CLOSE, MAGNITUDE, TIMESTAMP
from fart.model.prepare_datasets import prepare_datasets, train_test_split

CSV_HEADER = f"{TIMESTAMP},{CLOSE}\n"


def _write_candle_csv(path: Path, num_rows: int = 60) -> None:
    lines = [CSV_HEADER]
    base_price = 100.0
    for i in range(num_rows):
        timestamp = 1_600_000_000_000 + i * 86_400_000
        price = base_price + i
        lines.append(f"{timestamp},{price}\n")
    path.write_text("".join(lines))


def test_train_test_split_builds_lag_windows() -> None:
    data = np.arange(20, dtype=np.float32)

    x_train, y_train, x_val, y_val, x_test, y_test = train_test_split(
        data=data, num_lags=3, train_size=0.6, val_size=0.2
    )

    # num_windows = 20 - 3 = 17, train_end = int(0.6 * 17) = 10,
    # val_end = 10 + int(0.2 * 17) = 13
    assert x_train.shape == (10, 3)
    assert y_train.shape == (10,)
    assert x_val.shape == (3, 3)
    assert y_val.shape == (3,)
    assert x_test.shape == (4, 3)
    assert y_test.shape == (4,)
    np.testing.assert_array_equal(x_train[0], [0, 1, 2])
    assert y_train[0] == 3
    np.testing.assert_array_equal(x_val[0], [10, 11, 12])
    assert y_val[0] == 13
    np.testing.assert_array_equal(x_test[0], [13, 14, 15])
    assert y_test[0] == 16


def test_train_test_split_raises_when_not_enough_data() -> None:
    data = np.arange(3, dtype=np.float32)

    with pytest.raises(ValueError):
        train_test_split(data=data, num_lags=5)


def test_prepare_datasets_returns_windowed_split(tmp_path: Path) -> None:
    filepath = tmp_path / "BTC-EUR-1d.csv"
    _write_candle_csv(filepath, num_rows=60)

    x_train, y_train, x_val, y_val, x_test, y_test = prepare_datasets(
        data_filepath=filepath,
        target=MAGNITUDE,
        num_lags=5,
        train_size=0.6,
        val_size=0.2,
    )

    assert x_train.shape[1] == 5
    assert x_train.shape[0] == y_train.shape[0]
    assert x_val.shape[0] == y_val.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    assert x_train.shape[0] > x_val.shape[0]
    assert x_train.shape[0] > x_test.shape[0]
    assert np.all(np.isfinite(x_train))
    assert np.all(np.isfinite(y_train))


def test_prepare_datasets_deduplicates_and_sorts_candles(tmp_path: Path) -> None:
    clean_path = tmp_path / "clean.csv"
    tampered_path = tmp_path / "tampered.csv"
    _write_candle_csv(clean_path, num_rows=60)
    _write_candle_csv(tampered_path, num_rows=60)

    # Inject a duplicate of the second row and shuffle it out of order, as
    # can happen with a resumed download.
    lines = tampered_path.read_text().splitlines(keepends=True)
    header, rows = lines[0], lines[1:]
    duplicate = rows[1]
    tampered_rows = [rows[0], rows[2], duplicate, rows[1], *rows[3:]]
    tampered_path.write_text(header + "".join(tampered_rows))

    clean = prepare_datasets(data_filepath=clean_path, target=MAGNITUDE, num_lags=5)
    tampered = prepare_datasets(
        data_filepath=tampered_path, target=MAGNITUDE, num_lags=5
    )

    clean_total = clean[0].shape[0] + clean[2].shape[0] + clean[4].shape[0]
    tampered_total = tampered[0].shape[0] + tampered[2].shape[0] + tampered[4].shape[0]

    assert tampered_total == clean_total


def test_prepare_datasets_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_datasets(
            data_filepath=tmp_path / "missing.csv", target=MAGNITUDE, num_lags=5
        )

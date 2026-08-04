from pathlib import Path

import pytest
from loguru import logger

from fart.model.train_model import prepare_training_data, train

CSV_HEADER = "Timestamp,Open,High,Low,Close,Volume\n"


def _write_candle_csv(path: Path, num_rows: int = 60) -> None:
    lines = [CSV_HEADER]
    base_price = 100.0
    for i in range(num_rows):
        timestamp = 1_600_000_000_000 + i * 86_400_000
        price = base_price + i
        lines.append(
            f"{timestamp},{price},{price + 1},{price - 1},{price},{1000 + i}\n"
        )
    path.write_text("".join(lines))


def test_prepare_training_data_returns_clean_split(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv")

    X_train, X_test, y_train, y_test = prepare_training_data(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval="1d",
    )

    assert X_train.null_count().sum_horizontal().item() == 0
    assert X_test.null_count().sum_horizontal().item() == 0
    assert not y_train.is_null().any()
    assert not y_test.is_null().any()

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[0] + X_test.shape[0] > 0
    assert X_train.shape[0] > X_test.shape[0]


def test_prepare_training_data_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_training_data(data_dir=tmp_path, market="BTC-EUR", interval="1d")


def test_prepare_training_data_filters_to_recent_months(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=400)

    X_train, X_test, y_train, y_test = prepare_training_data(
        data_dir=tmp_path, market="BTC-EUR", interval="1d", months=6
    )
    total_rows_filtered = X_train.shape[0] + X_test.shape[0]

    X_train_full, X_test_full, y_train_full, y_test_full = prepare_training_data(
        data_dir=tmp_path, market="BTC-EUR", interval="1d", months=None
    )
    total_rows_full = X_train_full.shape[0] + X_test_full.shape[0]

    assert total_rows_filtered < total_rows_full
    assert 100 <= total_rows_filtered <= 181
    assert total_rows_full > 300


def test_train_logs_prepared_shapes(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv")

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO")

    try:
        train(data_dir=tmp_path, market="BTC-EUR", interval="1d")
    finally:
        logger.remove(sink_id)

    assert any("X_train" in message for message in messages)

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from loguru import logger

from fart.model.nbeats_config import NBeatsConfig
from fart.model.nbeats_dataset import build_return_windows
from fart.model.nbeats_persistence import load_model
from fart.model.train_model import prepare_training_data, train
from fart.utils import get_latest_model_filepath

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


def test_prepare_training_data_deduplicates_and_sorts_candles(tmp_path: Path) -> None:
    clean_dir = tmp_path / "clean"
    tampered_dir = tmp_path / "tampered"
    clean_dir.mkdir()
    tampered_dir.mkdir()

    _write_candle_csv(clean_dir / "BTC-EUR-1d.csv")
    tampered_path = tampered_dir / "BTC-EUR-1d.csv"
    _write_candle_csv(tampered_path)

    # Inject a duplicate of the second row and shuffle it out of order, as
    # can happen with a resumed download.
    lines = tampered_path.read_text().splitlines(keepends=True)
    header, rows = lines[0], lines[1:]
    duplicate = rows[1]
    tampered_rows = [rows[0], rows[2], duplicate, rows[1], *rows[3:]]
    tampered_path.write_text(header + "".join(tampered_rows))

    clean = prepare_training_data(data_dir=clean_dir, market="BTC-EUR", interval="1d")
    tampered_result = prepare_training_data(
        data_dir=tampered_dir, market="BTC-EUR", interval="1d"
    )

    clean_total = sum(x.shape[0] for x in clean[:2])
    tampered_total = sum(x.shape[0] for x in tampered_result[:2])

    # The duplicate row is dropped, and the out-of-order rows are
    # re-sorted, so the result is identical to the untampered data despite
    # one extra (duplicate) line in the source CSV.
    assert tampered_total == clean_total


def test_prepare_training_data_missing_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prepare_training_data(data_dir=tmp_path, market="BTC-EUR", interval="1d")


def test_prepare_training_data_filters_to_recent_months(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=400)

    X_train, X_test, _, _ = prepare_training_data(
        data_dir=tmp_path, market="BTC-EUR", interval="1d", months=6
    )
    total_rows_filtered = X_train.shape[0] + X_test.shape[0]

    X_train_full, X_test_full, _, _ = prepare_training_data(
        data_dir=tmp_path, market="BTC-EUR", interval="1d", months=None
    )
    total_rows_full = X_train_full.shape[0] + X_test_full.shape[0]

    assert total_rows_filtered < total_rows_full
    assert 100 <= total_rows_filtered <= 181
    assert total_rows_full > 300


def test_train_logs_prepared_shapes(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=200)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO")

    try:
        train(
            data_dir=tmp_path,
            market="BTC-EUR",
            interval="1d",
            artifacts_dir=tmp_path / "artifacts",
            config=NBeatsConfig(epochs=2, num_stacks=1, num_blocks_per_stack=1),
            device=torch.device("cpu"),
        )
    finally:
        logger.remove(sink_id)

    assert any("X_train" in message for message in messages)


def test_train_fits_nbeats_and_returns_magnitude_confidence(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=200)

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO")

    try:
        magnitudes, confidences = train(
            data_dir=tmp_path,
            market="BTC-EUR",
            interval="1d",
            artifacts_dir=tmp_path / "artifacts",
            config=NBeatsConfig(epochs=2, num_stacks=1, num_blocks_per_stack=1),
            device=torch.device("cpu"),
        )
    finally:
        logger.remove(sink_id)

    assert isinstance(magnitudes, np.ndarray)
    assert isinstance(confidences, np.ndarray)
    assert magnitudes.shape == confidences.shape
    assert magnitudes.shape[0] > 0
    assert np.all(confidences > 0) and np.all(confidences < 1)
    assert any("X_train" in message for message in messages)


def test_train_saves_versioned_artifact_each_run(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=200)
    artifacts_dir = tmp_path / "artifacts"
    config = NBeatsConfig(epochs=1, num_stacks=1, num_blocks_per_stack=1, batch_size=8)

    train(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval="1d",
        artifacts_dir=artifacts_dir,
        config=config,
        device=torch.device("cpu"),
    )
    train(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval="1d",
        artifacts_dir=artifacts_dir,
        config=config,
        device=torch.device("cpu"),
    )

    saved = sorted(artifacts_dir.glob("*-BTC-EUR-1d-nbeats.pt"))
    assert len(saved) == 2
    assert saved[0] != saved[1]


def test_train_minibatches_with_small_batch_size(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=200)

    magnitudes, confidences = train(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval="1d",
        artifacts_dir=tmp_path / "artifacts",
        config=NBeatsConfig(
            epochs=1, num_stacks=1, num_blocks_per_stack=1, batch_size=4
        ),
        device=torch.device("cpu"),
    )

    assert magnitudes.shape[0] > 0
    assert np.all(np.isfinite(magnitudes))
    assert np.all(confidences > 0) and np.all(confidences < 1)


def test_train_saved_artifact_reproduces_output(tmp_path: Path) -> None:
    _write_candle_csv(tmp_path / "BTC-EUR-1d.csv", num_rows=200)
    artifacts_dir = tmp_path / "artifacts"
    config = NBeatsConfig(epochs=1, num_stacks=1, num_blocks_per_stack=1, batch_size=8)

    magnitudes, confidences = train(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval="1d",
        artifacts_dir=artifacts_dir,
        config=config,
        device=torch.device("cpu"),
    )

    artifact_path = get_latest_model_filepath(artifacts_dir, "BTC-EUR", "1d")
    loaded_model, loaded_config = load_model(artifact_path)

    _, _, y_train, y_test = prepare_training_data(
        data_dir=tmp_path, market="BTC-EUR", interval="1d"
    )
    n_train = y_train.shape[0]
    close_prices = pl.concat([y_train, y_test])
    X_all, _y_all = build_return_windows(close_prices, loaded_config.lookback)
    n_train_windows = max(0, n_train - loaded_config.lookback - 1)
    X_test_windows = X_all[n_train_windows:]

    loaded_model.eval()
    with torch.no_grad():
        output = loaded_model(X_test_windows)
    mu, log_sigma = output.unbind(-1)
    loaded_magnitudes = mu.numpy()
    loaded_confidences = (1 / (1 + log_sigma.exp())).numpy()

    np.testing.assert_allclose(loaded_magnitudes, magnitudes, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(loaded_confidences, confidences, rtol=1e-5, atol=1e-6)

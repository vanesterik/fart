from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from fart.downloader import Downloader


def _make_downloader(
    tmp_path: Path,
    mock_bitvavo: MagicMock,
    market: str = "BTC-EUR",
    known_markets: list[str] | None = None,
) -> Downloader:
    mock_bitvavo.return_value.markets.return_value = [
        {"market": m}
        for m in (known_markets if known_markets is not None else [market])
    ]
    return Downloader(
        data_dir=tmp_path,
        market=market,
        interval="1d",
        api_key="test-api-key",
        api_secret="test-api-secret",
    )


@patch("fart.downloader.Bitvavo")
def test_downloader_stores_configuration_and_computes_filepath(
    mock_bitvavo: MagicMock, tmp_path: Path
) -> None:
    downloader = _make_downloader(tmp_path, mock_bitvavo, market="BTC-EUR")

    assert downloader._filepath == tmp_path / "BTC-EUR-1d.csv"
    assert tmp_path.exists()


@patch("fart.downloader.Bitvavo")
def test_downloader_passes_api_credentials_to_client(
    mock_bitvavo: MagicMock, tmp_path: Path
) -> None:
    _make_downloader(tmp_path, mock_bitvavo)

    mock_bitvavo.assert_called_once_with(
        {"APIKEY": "test-api-key", "APISECRET": "test-api-secret"}
    )


@patch("fart.downloader.Bitvavo")
def test_downloader_unknown_market_raises(
    mock_bitvavo: MagicMock, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="ETH-EUR"):
        _make_downloader(
            tmp_path, mock_bitvavo, market="ETH-EUR", known_markets=["BTC-EUR"]
        )


@pytest.mark.parametrize(
    "interval,last_candle_timestamp,expected_start_timestamp",
    [
        ("1m", 1_000_000, 1_000_000 + 60_000),
        ("5m", 1_000_000, 1_000_000 + 5 * 60_000),
        ("1h", 1_000_000, 1_000_000 + 3_600_000),
        ("1d", 1_000_000, 1_000_000 + 86_400_000),
    ],
)
@patch("fart.downloader.Bitvavo")
def test_downloader_resumes_one_interval_past_last_cached_candle(
    mock_bitvavo: MagicMock,
    tmp_path: Path,
    interval: str,
    last_candle_timestamp: int,
    expected_start_timestamp: int,
) -> None:
    mock_bitvavo.return_value.markets.return_value = [{"market": "BTC-EUR"}]
    downloader = Downloader(
        data_dir=tmp_path,
        market="BTC-EUR",
        interval=interval,
        api_key="test-api-key",
        api_secret="test-api-secret",
    )

    last_candle = (last_candle_timestamp, 1.0, 1.0, 1.0, 1.0, 1.0)
    start_timestamp = downloader._determine_start_timestamp([last_candle])

    assert start_timestamp == expected_start_timestamp


@patch("fart.downloader.Bitvavo")
def test_downloader_resumes_from_launch_timestamp_without_cached_data(
    mock_bitvavo: MagicMock, tmp_path: Path
) -> None:
    downloader = _make_downloader(tmp_path, mock_bitvavo, market="BTC-EUR")

    assert downloader._determine_start_timestamp([]) == 1552089600000


@patch("fart.downloader.Bitvavo")
def test_downloader_logs_configuration_without_leaking_secrets(
    mock_bitvavo: MagicMock, tmp_path: Path
) -> None:
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="INFO")

    try:
        _make_downloader(tmp_path, mock_bitvavo, market="BTC-EUR")
    finally:
        logger.remove(sink_id)

    logged = "\n".join(messages)
    assert "BTC-EUR" in logged
    assert "test-api-key" not in logged
    assert "test-api-secret" not in logged

from pathlib import Path

from fart.utils import get_candle_filepath


def test_get_candle_filepath() -> None:
    filepath = get_candle_filepath(
        data_dir=Path("/tmp/fart-test-data"),
        market="BTC-EUR",
        interval="1d",
    )

    assert filepath == Path("/tmp/fart-test-data/BTC-EUR-1d.csv")


def test_get_candle_filepath_different_market_and_interval() -> None:
    filepath = get_candle_filepath(
        data_dir=Path("/tmp/fart-test-data"),
        market="ETH-EUR",
        interval="1h",
    )

    assert filepath == Path("/tmp/fart-test-data/ETH-EUR-1h.csv")

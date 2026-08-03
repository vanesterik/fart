from pathlib import Path

from fart.settings import Interval, Settings
from fart.utils import get_candle_filepath


def test_get_candle_filepath() -> None:
    settings = Settings(
        data_dir=Path("/tmp/fart-test-data"),
        market="BTC-EUR",
        interval=Interval.ONE_DAY,
    )

    filepath = get_candle_filepath(settings)

    assert filepath == Path("/tmp/fart-test-data/BTC-EUR-1d.csv")


def test_get_candle_filepath_different_market_and_interval() -> None:
    settings = Settings(
        data_dir=Path("/tmp/fart-test-data"),
        market="ETH-EUR",
        interval=Interval.ONE_HOUR,
    )

    filepath = get_candle_filepath(settings)

    assert filepath == Path("/tmp/fart-test-data/ETH-EUR-1h.csv")

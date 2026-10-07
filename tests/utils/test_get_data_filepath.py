from pathlib import Path

from fartt.utils import get_data_filepath


def test_get_data_filepath() -> None:
    filepath = get_data_filepath(
        data_dir=Path("/tmp/fartt-test-data"),
        market="BTC-EUR",
        interval="1d",
    )

    assert filepath == Path("/tmp/fartt-test-data/BTC-EUR-1d.csv")


def test_get_data_filepath_different_market_and_interval() -> None:
    filepath = get_data_filepath(
        data_dir=Path("/tmp/fartt-test-data"),
        market="ETH-EUR",
        interval="1h",
    )

    assert filepath == Path("/tmp/fartt-test-data/ETH-EUR-1h.csv")

from datetime import datetime, timezone
from pathlib import Path

from fart.utils import get_model_filepath


def test_get_model_filepath() -> None:
    timestamp = datetime(2026, 8, 4, 14, 47, 49, 32031, tzinfo=timezone.utc)

    filepath = get_model_filepath(
        artifacts_dir=Path("/tmp/fart-test-artifacts"),
        market="BTC-EUR",
        interval="1d",
        timestamp=timestamp,
    )

    assert filepath == Path(
        "/tmp/fart-test-artifacts/20260804T144749032031Z-BTC-EUR-1d.pt"
    )


def test_get_model_filepath_different_market_and_interval() -> None:
    timestamp = datetime(2026, 1, 1, 0, 0, 0, 0, tzinfo=timezone.utc)

    filepath = get_model_filepath(
        artifacts_dir=Path("/tmp/fart-test-artifacts"),
        market="ETH-EUR",
        interval="1h",
        timestamp=timestamp,
    )

    assert filepath == Path(
        "/tmp/fart-test-artifacts/20260101T000000000000Z-ETH-EUR-1h.pt"
    )

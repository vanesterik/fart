import tempfile
from pathlib import Path

import pytest

from fart.utils import get_latest_model_filepath


def test_get_latest_model_filepath_picks_max_by_filename() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        older = Path(temp_dir) / "20260101T000000000000Z-BTC-EUR-1d.pt"
        newer = Path(temp_dir) / "20260804T144749032031Z-BTC-EUR-1d.pt"
        other_market = Path(temp_dir) / "20261231T235959999999Z-ETH-EUR-1d.pt"

        older.touch()
        newer.touch()
        other_market.touch()

        assert get_latest_model_filepath(Path(temp_dir), "BTC-EUR", "1d") == newer


def test_get_latest_model_filepath_no_matches_raises() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            get_latest_model_filepath(Path(temp_dir), "BTC-EUR", "1d")

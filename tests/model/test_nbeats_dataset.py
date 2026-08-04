import polars as pl
import pytest
import torch

from fart.model.nbeats_dataset import build_return_windows


def test_build_return_windows_shapes_and_values() -> None:
    close_prices = pl.Series([100.0, 101.0, 99.0, 102.0, 104.0, 103.0])

    X, y = build_return_windows(close_prices, lookback=2)

    assert X.shape == (3, 2)
    assert y.shape == (3,)

    expected_returns = torch.tensor(
        [
            0.01,
            -0.019801980198019802,
            0.030303030303030304,
            0.0196078431372549,
            -0.009615384615384616,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(X[0], expected_returns[0:2], atol=1e-6)
    assert torch.allclose(X[1], expected_returns[1:3], atol=1e-6)
    assert torch.allclose(X[2], expected_returns[2:4], atol=1e-6)
    assert torch.allclose(y, expected_returns[2:5], atol=1e-6)


def test_build_return_windows_raises_when_too_few_prices() -> None:
    close_prices = pl.Series([100.0, 101.0, 102.0])

    with pytest.raises(ValueError):
        build_return_windows(close_prices, lookback=5)

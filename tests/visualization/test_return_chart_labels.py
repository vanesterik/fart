from datetime import UTC, datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from fartt.constants import CANDLE_RETURN, DATETIME
from fartt.visualization.candle_returns import plot_candle_returns
from fartt.visualization.predicted_vs_actual_scatter import predicted_vs_actual_scatter

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def no_show(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    plt.close("all")


# Plain "Return" also labels the per-trade return chart (trade_returns.py),
# so charts of a candle's return must say "Candle Return".


def test_candle_returns_chart_labels_the_candle_return() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    df = pl.DataFrame(
        {
            DATETIME: [start + timedelta(hours=h) for h in range(4)],
            CANDLE_RETURN: [None, 0.01, -0.02, 0.005],
        }
    )

    plot_candle_returns(df)

    assert plt.gcf().axes[0].get_ylabel() == "Candle Return"


def test_prediction_scatter_labels_the_candle_return() -> None:
    values = np.array([0.01, -0.02, 0.005])

    predicted_vs_actual_scatter(values, values)

    ax = plt.gcf().axes[0]
    assert ax.get_xlabel() == "Actual Candle Return"
    assert ax.get_ylabel() == "Predicted Candle Return"
    assert ax.get_title() == "Predicted vs. Actual Candle Return (Test Set)"

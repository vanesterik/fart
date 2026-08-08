from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from fart.constants import HONOLULU_BLUE
from fart.visualization.diverging_bar_chart import plot_diverging_bars


def plot_trade_returns(
    returns: Sequence[float],
    profits: Sequence[float],
    initial_capital: float = 500,
) -> None:
    """
    Plot `calculate_trade_returns`'s per-trade returns and profits together:
    a cumulative-capital line (`initial_capital` plus running `profits`) on
    the left axis, with each trade's `returns` as diverging bars on the
    right axis, both against a shared trade-number x-axis -- so the overall
    equity trajectory and which individual trades drove it are readable in
    one glance.

    Parameters
    ----------
    - returns (Sequence[float]): Per-trade net returns, as fractions (see
      `calculate_trade_returns`).
    - profits (Sequence[float]): Per-trade currency profits, compounding
      `initial_capital` across trades in order (see
      `calculate_trade_returns`).
    - initial_capital (float): Starting capital `profits` compounded from.

    """
    trade_number = np.arange(1, len(profits) + 1)
    capital = initial_capital + np.cumsum(profits)

    _, ax_capital = plt.subplots(  # pyright: ignore[reportUnknownMemberType] -- pyplot.subplots' **fig_kw is untyped upstream
        figsize=(16, 4), constrained_layout=True
    )
    ax_capital.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
        trade_number, capital, color=HONOLULU_BLUE, linewidth=1.2
    )
    ax_capital.set_xlabel(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_xlabel's **kwargs is untyped upstream
        "Trade"
    )
    ax_capital.set_ylabel(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_ylabel's **kwargs is untyped upstream
        "Capital"
    )
    ax_capital.set_title(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_title's **kwargs is untyped upstream
        "Trade Returns and Cumulative Profit"
    )

    ax_return = ax_capital.twinx()  # pyright: ignore[reportUnknownMemberType] -- Axes.twinx's return type is untyped upstream
    plot_diverging_bars(ax_return, trade_number, returns)
    ax_return.set_ylabel(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_ylabel's **kwargs is untyped upstream
        "Return"
    )

    # Draw the capital line above the return bars, and hide its axes
    # background so the bars underneath stay visible.
    ax_capital.set_zorder(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_zorder's **kwargs is untyped upstream
        ax_return.get_zorder() + 1  # pyright: ignore[reportUnknownMemberType] -- Axes.get_zorder's return type is untyped upstream
    )
    ax_capital.patch.set_visible(False)

    plt.show()  # pyright: ignore[reportUnknownMemberType] -- pyplot.show is untyped upstream

import matplotlib.pyplot as plt
import polars as pl

from fart.constants import DATETIME, DOVE_GREY, HONOLULU_BLUE, MAGNITUDE


def plot_magnitude(df: pl.DataFrame) -> None:
    """
    Plot the Magnitude column (signed percent change of Close between
    consecutive candles) over time, with a zero reference line to make
    the direction of each move easy to read.

    Parameters
    ----------
    - df (pl.DataFrame): Candle data with `Datetime` and `Magnitude`
      columns (see `features/calculate_magnitude.py`).

    """
    _, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType] -- pyplot.subplots' **fig_kw is untyped upstream
        figsize=(16, 4), constrained_layout=True
    )
    ax.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
        df[DATETIME], df[MAGNITUDE], color=HONOLULU_BLUE, linewidth=0.8
    )
    ax.axhline(  # pyright: ignore[reportUnknownMemberType] -- Axes.axhline's **kwargs is untyped upstream
        0, color=DOVE_GREY, linewidth=0.8, linestyle="--"
    )
    ax.set_title(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_title's **kwargs is untyped upstream
        "Magnitude of Close Price Movements"
    )
    ax.set_ylabel(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_ylabel's **kwargs is untyped upstream
        MAGNITUDE
    )

    plt.show()  # pyright: ignore[reportUnknownMemberType] -- pyplot.show is untyped upstream

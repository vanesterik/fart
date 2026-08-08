import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from fart.constants import IMPERIAL_RED_MAIN, PERSIAN_GREEN_MAIN


def plot_diverging_bars(
    ax: Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    width: float = 0.8,
) -> None:
    """
    Draw a diverging bar chart onto `ax`: bars rise above zero for positive
    `y` values and fall below zero for negative ones, colored by direction,
    so magnitude and direction are readable at a glance. Draws onto an
    existing `Axes` rather than creating its own figure, so callers can
    compose it into a larger figure (e.g. a twin axis).

    Parameters
    ----------
    - ax (Axes): Axes to draw the bars onto.
    - x (npt.ArrayLike): Bar positions.
    - y (npt.ArrayLike): Bar heights; sign determines color.
    - width (float): Bar width, in the same units as `x`.

    """
    values = np.asarray(y, dtype=np.float64)
    colors = [
        PERSIAN_GREEN_MAIN if value >= 0 else IMPERIAL_RED_MAIN for value in values
    ]
    ax.bar(  # pyright: ignore[reportUnknownMemberType] -- Axes.bar's **kwargs is untyped upstream
        x, values, width=width, color=colors, edgecolor="none"
    )

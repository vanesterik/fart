import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from fart.constants import HONOLULU_BLUE


def plot_confidence_calibration(
    confidence: npt.NDArray[np.float64],
    error: npt.NDArray[np.float64],
) -> None:
    """
    Plot N-BEATS confidence against actual absolute error, one point per
    test-set window, to check whether confidence is a useful calibration
    signal. Two panels are shown side by side: the full 0-1 confidence
    scale (to see how much of that range the model actually uses) and a
    panel zoomed to the observed confidence range (to check for structure
    within it). Error is log-scaled since it typically spans several
    orders of magnitude.

    Parameters
    ----------
    - confidence (npt.NDArray[np.float64]): Per-window confidence scores,
      squashed to (0, 1).
    - error (npt.NDArray[np.float64]): Per-window absolute error between
      predicted and actual return.

    """
    pearson_r = np.corrcoef(confidence, error)[0, 1]

    fig, (ax_full, ax_zoom) = plt.subplots(  # pyright: ignore[reportUnknownMemberType] -- pyplot.subplots' **fig_kw is untyped upstream
        1, 2, figsize=(14, 5), constrained_layout=True
    )

    for ax in (ax_full, ax_zoom):
        ax.scatter(  # pyright: ignore[reportUnknownMemberType] -- Axes.scatter's **kwargs is untyped upstream
            confidence, error, s=16, alpha=0.45, color=HONOLULU_BLUE, edgecolors="none"
        )
        ax.set_yscale("log")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Absolute error (log scale)")

    ax_full.set_xlim(0, 1)
    ax_full.set_title("Full confidence scale")

    margin = (confidence.max() - confidence.min()) * 0.08
    ax_zoom.set_xlim(confidence.min() - margin, confidence.max() + margin)
    ax_zoom.set_title("Zoomed to observed range")

    fig.suptitle(  # pyright: ignore[reportUnknownMemberType] -- Figure.suptitle's **kwargs is untyped upstream
        f"Confidence vs. actual error  (n={len(confidence)}, Pearson r={pearson_r:.3f})"
    )
    plt.show()  # pyright: ignore[reportUnknownMemberType] -- pyplot.show is untyped upstream

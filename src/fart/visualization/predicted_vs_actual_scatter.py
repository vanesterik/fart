import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from fart.constants import BLACK, IMPERIAL_RED_MAIN, PERSIAN_GREEN_MAIN


def predicted_vs_actual_scatter(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Scatter a model's test-set predictions against realized values, one
    point per sample, to visually assess whether the model tracks actual
    magnitude or has collapsed toward a near-constant prediction -- a
    failure mode RMSE/accuracy summary numbers alone can mask. A dashed
    `y = x` line marks a perfect prediction. Points are colored by
    whether the prediction's sign matches the actual value's sign, using
    the same `sign(y_pred) == sign(y_test)` check as
    `evaluate_model.py::calculate_accuracy`, so the color split is
    numerically consistent with the reported directional accuracy.

    Parameters
    ----------
    - y_test (np.ndarray): Realized test-period values.
    - y_pred (np.ndarray): Predicted values for the test period, same
      length and order as `y_test`.

    """
    if len(y_pred) != len(y_test):
        raise ValueError(
            f"y_pred and y_test must be the same length: got "
            f"{len(y_pred)} predicted vs. {len(y_test)} test values."
        )

    y_test = y_test.reshape(-1)
    y_pred = y_pred.reshape(-1)
    correct_direction = np.sign(y_pred) == np.sign(y_test)
    colors = [
        PERSIAN_GREEN_MAIN if correct else IMPERIAL_RED_MAIN
        for correct in correct_direction
    ]

    _, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType] -- pyplot.subplots' **fig_kw is untyped upstream
        figsize=(8, 8), constrained_layout=True
    )
    limit = float(np.abs(np.concatenate([y_test, y_pred])).max()) * 1.1
    ax.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
        [-limit, limit], [-limit, limit], color=BLACK, linestyle="--", linewidth=1
    )
    ax.scatter(  # pyright: ignore[reportUnknownMemberType] -- Axes.scatter's **kwargs is untyped upstream
        y_test, y_pred, c=colors, alpha=0.6, edgecolors="none"
    )
    ax.set_xlim(-limit, limit)  # pyright: ignore[reportUnknownMemberType] -- Axes.set_xlim's **kwargs is untyped upstream
    ax.set_ylim(-limit, limit)  # pyright: ignore[reportUnknownMemberType] -- Axes.set_ylim's **kwargs is untyped upstream
    ax.set_aspect("equal", adjustable="box")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_aspect's **kwargs is untyped upstream
    ax.set_xlabel("Actual Magnitude")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_xlabel's **kwargs is untyped upstream
    ax.set_ylabel("Predicted Magnitude")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_ylabel's **kwargs is untyped upstream
    ax.set_title("Predicted vs. Actual Magnitude (Test Set)")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_title's **kwargs is untyped upstream
    ax.legend(  # pyright: ignore[reportUnknownMemberType] -- Axes.legend's **kwargs is untyped upstream
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=PERSIAN_GREEN_MAIN,
                label="Correct direction",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=IMPERIAL_RED_MAIN,
                label="Wrong direction",
            ),
        ],
        loc="lower right",
    )

    plt.show()  # pyright: ignore[reportUnknownMemberType] -- pyplot.show is untyped upstream

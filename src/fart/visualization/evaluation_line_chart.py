import numpy as np
from matplotlib import pyplot as plt

from fart.constants import BLACK, HONOLULU_BLUE, IMPERIAL_RED_MAIN


def evaluation_line_chart(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
) -> None:
    """
    Plot held-out test data and a model's predictions over the test period
    on one chart, to visually evaluate how closely predictions track
    realized values out of sample. If `y_train` is given, it's plotted
    too, immediately preceding the test period, with a dash-dot vertical
    line marking the train/test split; if omitted, only `y_test` and
    `y_pred` are plotted, starting at x=0. Test data is dashed, matching
    training data's color; predictions are a thin solid line in a
    distinct color.

    Parameters
    ----------
    - y_test (np.ndarray): Test-period values, in chronological order.
    - y_pred (np.ndarray): Predicted values for the test period, same
      length and order as `y_test`.
    - y_train (Optional[np.ndarray]): Training-period values immediately
      preceding `y_test`, in chronological order. If omitted, only
      `y_test` and `y_pred` are plotted.

    """
    if len(y_pred) != len(y_test):
        raise ValueError(
            f"y_pred and y_test must be the same length: got "
            f"{len(y_pred)} predicted vs. {len(y_test)} test values."
        )

    train_offset = len(y_train) if y_train is not None else 0
    test_x = np.arange(train_offset, train_offset + len(y_test))

    _, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType] -- pyplot.subplots' **fig_kw is untyped upstream
        figsize=(14, 6), constrained_layout=True
    )
    if y_train is not None:
        train_x = np.arange(len(y_train))
        ax.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
            train_x, y_train, color=HONOLULU_BLUE, linewidth=1, label="Training data"
        )
        ax.axvline(  # pyright: ignore[reportUnknownMemberType] -- Axes.axvline's **kwargs is untyped upstream
            train_offset - 0.5, color=BLACK, linestyle="-.", linewidth=1
        )
    ax.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
        test_x,
        y_test,
        color=HONOLULU_BLUE,
        linewidth=1,
        linestyle="--",
        label="Test data",
    )
    ax.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
        test_x, y_pred, color=IMPERIAL_RED_MAIN, linewidth=1, label="Predicted data"
    )
    ax.legend(loc="lower right")  # pyright: ignore[reportUnknownMemberType] -- Axes.legend's **kwargs is untyped upstream
    ax.set_title(  # pyright: ignore[reportUnknownMemberType] -- Axes.set_title's **kwargs is untyped upstream
        "Training, Test, and Predicted Data"
        if y_train is not None
        else "Test and Predicted Data"
    )

    plt.show()  # pyright: ignore[reportUnknownMemberType] -- pyplot.show is untyped upstream

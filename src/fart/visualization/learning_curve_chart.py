import matplotlib.pyplot as plt

from fart.constants import IMPERIAL_RED_MAIN, PERSIAN_GREEN_MAIN


def learning_curve_chart(cv_results: list[dict[str, float]]) -> None:
    """
    Plot a train-vs-validation learning curve (loss per epoch) for every
    cross-validation fold in `cv_results` on one chart, to check for
    overfitting -- a validation line drifting away from its training line
    means the model is memorizing that fold's training data rather than
    generalizing. All folds share the same two colors (train vs.
    validation); folds are distinguished only by their own trajectory,
    not by a separate color per fold.

    Parameters
    ----------
    - cv_results (list[dict[str, float]]): Per-(fold, epoch) records from
      `fart/model/train_model.py::train_model`, each with `fold`, `epoch`,
      `train_loss`, `val_loss`. Raises if empty -- pass a model trained
      with `num_splits > 1` to get per-fold history.

    """
    if not cv_results:
        raise ValueError(
            "cv_results is empty -- call train_model() with num_splits > 1 "
            "to get per-fold history to plot."
        )

    folds = sorted({record["fold"] for record in cv_results})

    _, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType] -- pyplot.subplots' **fig_kw is untyped upstream
        figsize=(10, 6), constrained_layout=True
    )

    for i, fold in enumerate(folds):
        fold_records = sorted(
            (record for record in cv_results if record["fold"] == fold),
            key=lambda record: record["epoch"],
        )
        epochs = [record["epoch"] for record in fold_records]
        train_loss = [record["train_loss"] for record in fold_records]
        val_loss = [record["val_loss"] for record in fold_records]

        ax.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
            epochs,
            train_loss,
            color=PERSIAN_GREEN_MAIN,
            alpha=0.7,
            label="Train loss" if i == 0 else None,
        )
        ax.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
            epochs,
            val_loss,
            color=IMPERIAL_RED_MAIN,
            alpha=0.7,
            label="Validation loss" if i == 0 else None,
        )

    ax.set_title("Learning Curve: Train vs. Validation Loss")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_title's **kwargs is untyped upstream
    ax.set_xlabel("Epoch")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_xlabel's **kwargs is untyped upstream
    ax.set_ylabel("Loss")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_ylabel's **kwargs is untyped upstream
    ax.legend()  # pyright: ignore[reportUnknownMemberType] -- Axes.legend's **kwargs is untyped upstream

    plt.show()  # pyright: ignore[reportUnknownMemberType] -- pyplot.show is untyped upstream

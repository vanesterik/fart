import matplotlib.pyplot as plt

from fart.constants import IMPERIAL_RED_MAIN, PERSIAN_GREEN_MAIN


def learning_curve_chart(history: list[dict[str, float]]) -> None:
    """
    Plot a train-vs-validation learning curve (loss per epoch), to check
    for overfitting -- a validation line drifting away from its training
    line means the model is memorizing the training data rather than
    generalizing.

    Parameters
    ----------
    - history (list[dict[str, float]]): Per-epoch records from
      `fart/model/train_model.py::train_model`, each with `epoch`,
      `train_loss`, `val_loss`. Raises if empty -- pass a model trained
      with `x_val`/`y_val` given to get per-epoch history.

    """
    if not history:
        raise ValueError(
            "history is empty -- call train_model() with x_val/y_val given "
            "to get per-epoch history to plot."
        )

    records = sorted(history, key=lambda record: record["epoch"])
    epochs = [record["epoch"] for record in records]
    train_loss = [record["train_loss"] for record in records]
    val_loss = [record["val_loss"] for record in records]

    _, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType] -- pyplot.subplots' **fig_kw is untyped upstream
        figsize=(10, 6), constrained_layout=True
    )
    ax.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
        epochs, train_loss, color=PERSIAN_GREEN_MAIN, label="Train loss"
    )
    ax.plot(  # pyright: ignore[reportUnknownMemberType] -- Axes.plot's **kwargs is untyped upstream
        epochs, val_loss, color=IMPERIAL_RED_MAIN, label="Validation loss"
    )

    ax.set_title("Learning Curve: Train vs. Validation Loss")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_title's **kwargs is untyped upstream
    ax.set_xlabel("Epoch")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_xlabel's **kwargs is untyped upstream
    ax.set_ylabel("Loss")  # pyright: ignore[reportUnknownMemberType] -- Axes.set_ylabel's **kwargs is untyped upstream
    ax.legend()  # pyright: ignore[reportUnknownMemberType] -- Axes.legend's **kwargs is untyped upstream

    plt.show()  # pyright: ignore[reportUnknownMemberType] -- pyplot.show is untyped upstream

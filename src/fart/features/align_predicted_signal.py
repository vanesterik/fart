import numpy as np
import numpy.typing as npt


def align_predicted_signal(y_pred: npt.ArrayLike) -> np.ndarray:
    """
    Shift a same-index model prediction into the one-tick-ahead trading
    signal `calculate_trade_returns`'s `predicted_magnitudes` parameter
    expects.

    A model trained via `prepare_datasets` predicts `y[i]` from a lag
    window ending right before it, so `y_pred[i]` is causally available
    at the same moment `y[i - 1]` closes, forecasting `y[i]`.
    `calculate_trade_returns` instead expects `predicted_magnitudes[i]`
    to be a forecast of `magnitudes[i + 1]`, available at the close of
    candle `i` (its own documented convention, e.g.
    `magnitude_series.shift(-1)`). Reindexing `y_pred` from "forecasts
    index i" to "available at index i, forecasts index i + 1" means
    shifting it back by one position; the last position has no later
    prediction to fill it, so it's set to `NaN`, which
    `calculate_trade_returns` already treats as no signal.

    Parameters
    ----------
    - y_pred (npt.ArrayLike): Same-index model predictions, e.g.
      `y_test_pred` from `evaluate_model`, where `y_pred[i]` forecasts
      `y_test[i]`.

    Returns
    -------
    - np.ndarray: `y_pred` shifted back by one index with a trailing
      `NaN`, suitable for `calculate_trade_returns`'s
      `predicted_magnitudes` parameter.

    """
    values = np.asarray(y_pred, dtype=np.float64)

    return np.concatenate([values[1:], [np.nan]])

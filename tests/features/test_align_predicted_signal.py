import math

import pytest

from fart.features.align_predicted_signal import align_predicted_signal
from fart.features.calculate_trade_returns import calculate_trade_returns


def test_align_predicted_signal_shifts_back_by_one() -> None:
    y_pred = [0.001, 0.05, 0.001, -0.01]

    aligned = align_predicted_signal(y_pred)

    assert aligned[:-1] == pytest.approx(y_pred[1:])
    assert math.isnan(aligned[-1])
    assert len(aligned) == len(y_pred)


def test_align_predicted_signal_recovers_the_move_a_same_index_predictor_would_miss() -> (
    None
):
    # y_pred[i] is a (perfect) same-index forecast of magnitudes[i] -- the
    # shape evaluate_model's y_test_pred actually has. Passed to
    # calculate_trade_returns unshifted, entry fires on the spike itself but
    # return capture starts one candle late and misses it; aligning first
    # recovers the spike, matching calculate_trade_returns's own documented
    # one-tick-ahead convention.
    magnitudes = [0.001, 0.001, 0.05, 0.001, -0.01, 0.001]
    y_pred = magnitudes

    unaligned_returns, _ = calculate_trade_returns(magnitudes, y_pred)
    aligned_returns, _ = calculate_trade_returns(
        magnitudes, align_predicted_signal(y_pred)
    )

    cost_factor = (1 - 0.0025) ** 2
    expected_aligned_return = (1 + 0.05) * (1 + 0.001) * cost_factor - 1
    assert aligned_returns == pytest.approx([expected_aligned_return])
    assert aligned_returns[0] > unaligned_returns[0]

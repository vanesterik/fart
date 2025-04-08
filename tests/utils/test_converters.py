from datetime import datetime


from fart.utils.converters import (
    calculate_profit_loss,
    convert_balance_data,
    convert_currency_data,
    convert_timestamp,
    create_period_timestamps,
    filter_trades_by_timestamp,
)
from fart.core.exchange import BalanceAsset, Trade


def test_convert_balance_data() -> None:
    balance = [
        BalanceAsset(symbol="BTC", available="1.5", inOrder="0.0"),
        BalanceAsset(symbol="ETH", available="2.0", inOrder="0.0"),
        BalanceAsset(symbol="XRP", available="invalid", inOrder="0.0"),
    ]
    result = convert_balance_data(balance)
    expected = [("BTC", 1.5), ("ETH", 2.0), ("XRP", None)]
    assert result == expected


def test_convert_currency_data() -> None:
    candle = (1609459200000, "10000", "10500", "9500", "10200", "1500")
    result = convert_currency_data(candle)
    expected = (10200.0, 0.02, 10500.0, 9500.0, 1500.0)
    assert result == expected


def test_convert_timestamp() -> None:
    date = datetime(2021, 1, 1, 0, 0, 0)
    result = convert_timestamp(date)
    expected = 1609455600000
    assert result == expected


def test_create_period_timestamps() -> None:
    result = create_period_timestamps(
        datetime(2023, 5, 15, 12, 0, 0)
    )  # Monday, May 15, 2023, 12:00:00

    expected_today = convert_timestamp(datetime(2023, 5, 15))
    expected_this_week = convert_timestamp(datetime(2023, 5, 15))
    expected_this_month = convert_timestamp(datetime(2023, 5, 1))
    expected_year_to_date = convert_timestamp(datetime(2023, 1, 1))

    expected = [
        expected_today,
        expected_this_week,
        expected_this_month,
        expected_year_to_date,
    ]

    assert len(result) == 4
    assert result == expected


def test_filter_trades_by_timestamp() -> None:
    trades = [
        create_mock_trade(1609459200000, "buy"),
        create_mock_trade(1609545600000, "sell"),
    ]
    timestamp = 1609459200000
    result = filter_trades_by_timestamp(trades, timestamp)
    expected = [trades[1]]
    assert result == expected


def test_calculate_profit_loss() -> None:
    trades = [
        create_mock_trade(1609459200000, "buy", "10000"),
        create_mock_trade(1609545600000, "sell", "10500"),
    ]
    result = calculate_profit_loss(trades)
    expected = 480.0  # (10500 * 1 - 10) - (10000 * 1 + 10)
    assert result == expected


def create_mock_trade(timestamp: int, side: str, price: str = "10000") -> Trade:
    return Trade(
        amount="1.0",
        clientOrderId="1",
        fee="10",
        feeCurrency="EUR",
        id="1",
        market="BTC-EUR",
        orderId="1",
        price=price,
        settled=True,
        side=side,
        taker=True,
        timestamp=timestamp,
    )

from datetime import datetime, time, timedelta
from typing import List

from fart.utils.dashboard import BalanceData, CurrencyData, ProfitLossData
from fart.utils.trader import BalanceAsset, Candle, Trade


def convert_balance_data(balance: List[BalanceAsset]) -> BalanceData:
    result = []

    for asset in balance:
        try:
            available = float(asset.available)
        except ValueError:
            available = None
        result.append((asset.symbol, available))

    return result


def convert_currency_data(candle: Candle) -> CurrencyData:
    open = float(candle[1])
    high = float(candle[2])
    low = float(candle[3])
    close = float(candle[4])
    volume = float(candle[5])

    return (
        close,  # Close price is the actual price
        (close - open) / open,  # Change is the percentage change from open to close
        high,
        low,
        volume,
    )


def convert_timestamp(date: datetime) -> int:
    return int(date.timestamp() * 1000)


def create_period_timestamps(now: datetime = datetime.now()) -> List[int]:
    today = datetime.combine(now.date(), time.min)
    this_week = today - timedelta(days=today.weekday())
    this_month = today.replace(day=1)
    year_to_date = today.replace(month=1, day=1)

    return [
        convert_timestamp(period)
        for period in (today, this_week, this_month, year_to_date)
    ]


def filter_trades_by_timestamp(trades: List[Trade], timestamp: int) -> List[Trade]:
    return list(filter(lambda x: x.timestamp > timestamp, trades))


def convert_profit_loss_data(trades: List[Trade]) -> ProfitLossData:
    (today, this_week, this_month, year_to_data) = create_period_timestamps()

    return (
        calculate_profit_loss(filter_trades_by_timestamp(trades, today)),
        calculate_profit_loss(filter_trades_by_timestamp(trades, this_week)),
        calculate_profit_loss(filter_trades_by_timestamp(trades, this_month)),
        calculate_profit_loss(filter_trades_by_timestamp(trades, year_to_data)),
        calculate_profit_loss(trades),
    )


def calculate_profit_loss(trades: List[Trade]) -> float:
    position = 0.0
    cost = 0.0
    revenue = 0.0
    profit_loss = 0.0

    for trade in trades:
        amount = float(trade.amount)
        price = float(trade.price)
        fee = float(trade.fee)

        if trade.side == "buy":
            position += amount
            cost += amount * price + fee
        elif trade.side == "sell":
            position -= amount
            revenue += amount * price - fee

    realized = revenue - cost

    # Calculate unrealized PnL for remaining position
    if position > 0:
        last_price = float(trades[-1].price)
        unrealized = position * last_price - (cost - revenue)
        profit_loss = realized + unrealized
    else:
        profit_loss = realized

    return profit_loss

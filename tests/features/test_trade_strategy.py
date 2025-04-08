import polars as pl

from fart.common.constants import CLOSE, TIMESTAMP, TRADE_SIGNAL
from fart.features.trade_strategy import TradeStrategy


def test_initial_capital() -> None:
    df = pl.DataFrame({TRADE_SIGNAL: [], CLOSE: [], TIMESTAMP: []})
    strategy = TradeStrategy(df, initial_capital=1000)
    assert strategy.initial_capital == 1000


def test_proceeds_no_trades() -> None:
    df = pl.DataFrame({TRADE_SIGNAL: [], CLOSE: [], TIMESTAMP: []})
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.proceeds == 1000


def test_total_return_no_trades() -> None:
    df = pl.DataFrame({TRADE_SIGNAL: [], CLOSE: [], TIMESTAMP: []})
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.total_return == 0.0


def test_single_trade() -> None:
    df = pl.DataFrame(
        {
            TRADE_SIGNAL: [1, -1],
            CLOSE: [100, 110],
            TIMESTAMP: ["2023-01-01", "2023-01-02"],
        }
    )
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.proceeds == 1100
    assert len(strategy.trades) == 2


def test_multiple_trades() -> None:
    df = pl.DataFrame(
        {
            TRADE_SIGNAL: [1, -1, 1, -1],
            CLOSE: [100, 110, 200, 210],
            TIMESTAMP: ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        }
    )
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.proceeds == 2100
    assert len(strategy.trades) == 2


def test_open_position_at_end() -> None:
    df = pl.DataFrame({TRADE_SIGNAL: [1], CLOSE: [100], TIMESTAMP: ["2023-01-01"]})
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.proceeds == 1000
    assert len(strategy.trades) == 2  # Open and close at the end

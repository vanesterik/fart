import polars as pl

from fart.constants import feature_names as fn
from fart.features.trade_strategy import TradeStrategy


def test_initial_capital() -> None:
    df = pl.DataFrame({fn.TRADE_SIGNAL: [], fn.CLOSE: [], fn.TIMESTAMP: []})
    strategy = TradeStrategy(df, initial_capital=1000)
    assert strategy.initial_capital == 1000


def test_proceeds_no_trades() -> None:
    df = pl.DataFrame({fn.TRADE_SIGNAL: [], fn.CLOSE: [], fn.TIMESTAMP: []})
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.proceeds == 1000


def test_total_return_no_trades() -> None:
    df = pl.DataFrame({fn.TRADE_SIGNAL: [], fn.CLOSE: [], fn.TIMESTAMP: []})
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.total_return == 0.0


def test_single_trade() -> None:
    df = pl.DataFrame(
        {
            fn.TRADE_SIGNAL: [1, -1],
            fn.CLOSE: [100, 110],
            fn.TIMESTAMP: ["2023-01-01", "2023-01-02"],
        }
    )
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.proceeds == 1100
    assert len(strategy.trades) == 2


def test_multiple_trades() -> None:
    df = pl.DataFrame(
        {
            fn.TRADE_SIGNAL: [1, -1, 1, -1],
            fn.CLOSE: [100, 110, 200, 210],
            fn.TIMESTAMP: ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        }
    )
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.proceeds == 1155
    assert len(strategy.trades) == 4


def test_open_position_at_end() -> None:
    df = pl.DataFrame(
        {fn.TRADE_SIGNAL: [1], fn.CLOSE: [100], fn.TIMESTAMP: ["2023-01-01"]}
    )
    strategy = TradeStrategy(df, initial_capital=1000)
    strategy.backtest()
    assert strategy.proceeds == 1000
    assert len(strategy.trades) == 2  # Open and close at the end

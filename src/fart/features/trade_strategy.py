from typing import Any, Dict, List, Tuple

import polars as pl

from fart.common.constants import BUY_CLASS, CLOSE, SELL_CLASS, TIMESTAMP, TRADE_SIGNAL


class TradeStrategy:
    def __init__(
        self,
        df: pl.DataFrame,
        initial_capital: float = 500,
        transaction_cost: float = 0.0,
    ) -> None:
        """
        Initialize the TradeStrategy class containing methods to backtest a
        trading strategy. The backtest is performed on a the passed DataFrame
        containing the necessary columns.

        Parameters
        ----------
        - df (pl.DataFrame): DataFrame containing the necessary columns.
        - initial_capital (float): Initial capital to start trading with.
        - transaction_cost (float): Transaction cost percentage per trade.

        """

        self._df = df
        self._initial_capital = initial_capital
        self._is_open_position = False
        self._proceeds = initial_capital
        self._shares = 0.0
        self._trades: List[Tuple[Any, Any, float, float, float]] = []
        self._transaction_cost = transaction_cost

    @property
    def initial_capital(self) -> float:
        """
        Returns the initial capital used to backtest the trading strategy.

        """
        return self._initial_capital

    @property
    def proceeds(self) -> float:
        """
        Returns the total proceeds of backtesting the trading strategy.

        """
        return self._proceeds

    @property
    def total_return(self) -> float:
        """
        Returns the total return of the trading strategy. The total return is
        calculated as the percentage difference between the total proceeds and
        the initial amount.

        """
        return (self._proceeds - self._initial_capital) / self._initial_capital

    @property
    def trades(self) -> List[Tuple[Any, Any, float, float, float]]:
        """
        Returns the list of trades made during the backtest. Each trade is a
        tuple containing the following elements: (action, timestamp, price,
        shares, proceeds).

        """
        return self._trades

    def backtest(self) -> None:
        """
        Backtest the trading strategy on the passed DataFrame. The trading
        strategy is based on the trade signal column in the DataFrame. The trade
        signal column should contain 1 for a buy signal and -1 for a sell
        signal.

        """

        # Iterate over each row in the DataFrame and process the row
        for row in self._df.rows(named=True):
            self._process_row(row)

        # Close the last open position
        self._close_last_open_position()

    def _process_row(self, row: Dict[str, Any]) -> None:
        """
        Process a row in the DataFrame. If it is a buy trade signal there is no
        open position, open a position. If it is a sell trade signal there is
        an open position, close the position.

        Parameters
        ----------
        - row (Dict[str, Any]]]): Row in the DataFrame

        """
        if row[TRADE_SIGNAL] == BUY_CLASS and not self._is_open_position:
            self._open_position(row)

        elif row[TRADE_SIGNAL] == SELL_CLASS and self._is_open_position:
            self._close_position(row)

    def _open_position(self, row: Dict[str, Any]) -> None:
        """
        Open a position at the current row. The position size is determined by
        the available proceeds and the transaction cost.

        Parameters
        ----------
        - row (Dict[str, Any]]): Row in the DataFrame

        """
        self._is_open_position = True
        entry_price = float(row[CLOSE])
        position_size = self._proceeds * (1 - self._transaction_cost)
        self._shares = position_size / entry_price
        self._proceeds -= position_size
        self._trades.append(
            (
                row[TIMESTAMP],
                row[TRADE_SIGNAL],
                entry_price,
                self._shares,
                self._proceeds,
            )
        )

    def _close_position(self, row: Dict[str, Any]) -> None:
        """
        Close the current open position at the current row. The position value
        is determined by the number of shares and the exit price.

        Parameters
        ----------
        - row (Dict[str, Any]]): Row in the DataFrame

        """

        self._is_open_position = False
        exit_price = float(row[CLOSE])
        position_value = self._shares * exit_price * (1 - self._transaction_cost)
        self._proceeds += position_value
        self._trades.append(
            (
                row[TIMESTAMP],
                row[TRADE_SIGNAL],
                exit_price,
                self._shares,
                self._proceeds,
            )
        )

    def _close_last_open_position(self) -> None:
        """
        Close the last open position if there is an open position at the end of
        the backtest.

        Parameters
        ----------
        - row (Dict[str, Any]]): Row in the DataFrame

        """
        if self._is_open_position:
            last_row = self._df.tail(1).row(0, named=True)
            self._close_position(last_row)

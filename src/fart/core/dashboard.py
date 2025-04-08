from datetime import datetime
from typing import List, Optional, Tuple

from babel.numbers import format_decimal, format_percent
from rich.columns import Columns
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel as BasePanel
from rich.table import Table as BaseTable
from rich.text import Text

from fart.common.constants import (
    BALANCE,
    CHANGE,
    DOVE_GREY,
    EUR,
    GREEN,
    HIGH,
    LAST_UPDATE,
    LOW,
    NOT_AVAILABLE,
    PRICE,
    PROFIT_LOSS,
    RED,
    THIS_MONTH,
    THIS_WEEK,
    TODAY,
    TOTAL,
    VOLUME,
    YEAR_TO_DATE,
)

BalanceData = Optional[
    List[
        Tuple[
            str,  # Symbol
            Optional[float],  # Available
        ]
    ]
]

CurrencyData = Optional[
    Tuple[
        float,  # Price
        float,  # Change
        float,  # High
        float,  # Low
        float,  # Volume
    ]
]

ProfitLossData = Optional[
    Tuple[
        float,  # Today
        float,  # This Week
        float,  # This Month
        float,  # Year to Date
        float,  # Total
    ]
]


class Dashboard:
    """
    Dashboard based on Rich classes with predefined style. The dashboard
    consists of a currency table, a balance table and a profit and loss table.
    Above these tables, the trades of the day will be rendered. This to give a
    full overview of the current state of the trading service.

    This class is reactive in rendering based on data provided. As such the
    initial initialization is done with placeholders.
    """

    def __init__(self) -> None:
        self._market: str = NOT_AVAILABLE
        self._interval: str = NOT_AVAILABLE
        self._balance: BalanceData = None
        self._currency: CurrencyData = None
        self._profit_loss: ProfitLossData = None
        self._console = Console()
        self._live: Live | None = None

    @property
    def market(self) -> str:
        return self._market

    @market.setter
    def market(self, market: str) -> None:
        self._market = market

    @property
    def interval(self) -> str:
        return self._interval

    @interval.setter
    def interval(self, interval: str) -> None:
        self._interval = interval

    @property
    def balance(self) -> BalanceData:
        return self._balance

    @balance.setter
    def balance(self, balance: BalanceData) -> None:
        self._balance = balance

    @property
    def currency(self) -> CurrencyData:
        return self._currency

    @currency.setter
    def currency(self, currency: CurrencyData) -> None:
        self._currency = currency

    @property
    def profit_loss(self) -> ProfitLossData:
        return self._profit_loss

    @profit_loss.setter
    def profit_loss(self, profit_loss: ProfitLossData) -> None:
        self._profit_loss = profit_loss

    def _generate_dashboard(self) -> RenderableType:
        return Group(
            Columns(
                [
                    Panel(
                        CurrencyTable(self._currency),
                        title=f"{self._market} ({self._interval})",
                    ),
                    Panel(
                        BalanceTable(self._balance),
                        title=BALANCE,
                    ),
                    Panel(
                        ProfitLossTable(self._profit_loss),
                        title=PROFIT_LOSS,
                    ),
                ],
            ),
            Text(
                f"{LAST_UPDATE}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                style=DOVE_GREY,
            ),
        )

    def initiate(self) -> None:
        self._console.clear()
        self._live = Live(
            self._generate_dashboard(),
            refresh_per_second=4,
            console=self._console,
        )
        self._live.start()

    def render(self) -> None:
        if self._live:
            self._live.update(self._generate_dashboard())


class Panel(BasePanel):
    """
    Panel based on Rich Panel with a title and border predefined style.

    Parameters
    ----------
    - renderable (RenderableType): Renderable object to display in the panel.
    - title (str): Title of the panel.

    """

    def __init__(
        self,
        renderable: RenderableType,
        title: str,
    ) -> None:
        super().__init__(
            border_style=DOVE_GREY,
            height=10,
            padding=(1, 1),
            renderable=renderable,
            title_align="left",
            title=title,
        )


class Table(BaseTable):
    """
    Table based on Rich Table with predefined style.

    Parameters
    ----------
    - pad_edge (bool): Whether to pad the edge of the table.

    """

    def __init__(self, pad_edge: bool = False) -> None:
        super().__init__(
            box=None,
            min_width=24,
            pad_edge=pad_edge,
            show_edge=False,
            show_header=False,
        )


class DecimalText(Text):
    """
    Decimal Text based on Rich Text with a decimal value.

    Parameters
    ----------
    - value (float): Decimal value to display.
    - currency (bool): Whether the value is a currencycurrency value.

    """

    def __init__(self, value: Optional[float] = None, currency: bool = False) -> None:
        super().__init__(
            (
                format_decimal(value, format="#.00000000" if currency else "#,##0.00")
                if value is not None
                else NOT_AVAILABLE
            ),
            justify="right",
            style=RED if value is not None and value < 0 else "",
        )


class PercentText(Text):
    """
    Percent Text based on Rich Text with a percent value.

    Parameters
    ----------
    - value (float): Percent value to display.

    """

    def __init__(self, value: Optional[float] = None) -> None:
        super().__init__(
            (
                format_percent(value, format="#.000%")
                if value is not None
                else NOT_AVAILABLE
            ),
            justify="right",
            style=(
                RED
                if value is not None and value < 0
                else GREEN if value is not None and value > 0 else ""
            ),
        )


class BalanceTable(Table):
    """
    Balance table based on custom Rich Table class with predefined style.

    Parameters
    ----------
    - balance (BalanceData): List of asset symbols and available amounts.

    """

    def __init__(self, balance: BalanceData = None) -> None:
        self._balance = balance if balance is not None else [(EUR, None)]

        super().__init__()

        self.add_column()
        self.add_column()

        for symbol, available in self._balance:
            self.add_row(
                symbol,
                DecimalText(available, currency=True if symbol != EUR else False),
            )


class CurrencyTable(Table):
    """
    Currency table based on custom Rich Table class with predefined style.

    Parameters
    ----------
    - currency (CurrencyData): Currency data containing price, change, high, low
      and volume.

    """

    def __init__(
        self,
        currency: CurrencyData = None,
    ) -> None:
        (
            self._price,
            self._change,
            self._high,
            self._low,
            self._volume,
        ) = (
            currency
            if currency is not None
            else (
                None,
                None,
                None,
                None,
                None,
            )
        )

        super().__init__()

        self.add_column()
        self.add_column()
        self.add_row(PRICE, DecimalText(self._price))
        self.add_row(CHANGE, PercentText(self._change))
        self.add_row()
        self.add_row(HIGH, DecimalText(self._high))
        self.add_row(LOW, DecimalText(self._low))
        self.add_row(VOLUME, DecimalText(self._volume, currency=True))


class ProfitLossTable(Table):
    """
    Profit and loss table based on custom Rich Table class with predefined
    style.

    Parameters
    ----------
    - profit_loss (ProfitLossData): Profit and loss data containing today, week,
      month, year and total.

    """

    def __init__(
        self,
        profit_loss: ProfitLossData = None,
    ) -> None:
        (
            self._today,
            self._week,
            self._month,
            self._year,
            self._total,
        ) = (
            profit_loss
            if profit_loss is not None
            else (
                None,
                None,
                None,
                None,
                None,
            )
        )

        super().__init__()

        self.add_column()
        self.add_column()
        self.add_row(TODAY, DecimalText(self._today))
        self.add_row(THIS_WEEK, DecimalText(self._week))
        self.add_row(THIS_MONTH, DecimalText(self._month))
        self.add_row(YEAR_TO_DATE, DecimalText(self._year))
        self.add_row()
        self.add_row(TOTAL, DecimalText(self._total))


class TransactionHistoryTable(Table):
    """
    Transaction history table based on custom Rich Table class with predefined
    style.

    Parameters
    ----------
    - transactions (List[Tuple[str, str, float, float]]): List of transaction
      details.

    """

    def __init__(self, transactions: List[Tuple[str, str, float, float]]) -> None:
        self._transactions = transactions

        super().__init__(pad_edge=True)

        self.add_column()
        self.add_column()
        self.add_column()
        self.add_column()

        for timestamp, action, amount, price in self._transactions:
            self.add_row(
                timestamp,
                action,
                DecimalText(amount),
                DecimalText(price),
            )

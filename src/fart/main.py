from os import getenv
from typing import Any, Dict

import click
from dotenv import find_dotenv, load_dotenv

from fart.core.dashboard import Dashboard
from fart.core.exchange import CandlesSubscription, Exchange
from fart.utils.converters import (
    convert_balance_data,
    convert_currency_data,
    convert_profit_loss_data,
)


@click.command()
@click.argument("market", type=click.STRING, default="BTC-EUR")
@click.argument("interval", type=click.STRING, default="1m")
def main(market: str, interval: str) -> None:

    # Set config of trader
    exchange.market = market
    exchange.interval = interval

    # Set config of dashboard
    dashboard.market = market
    dashboard.interval = interval

    # Initiate dashboard screen
    dashboard.initiate()
    update_dashboard(
        {
            "event": "candle",
            "market": market,
            "interval": interval,
            "candle": [exchange.last_candle],
        }
    )

    # Initiate trading service
    exchange.initiate(update_dashboard)
    exchange.wait_and_close()


def update_dashboard(candle_data: Dict[str, Any]) -> None:

    # Type and extract candle data
    candle = CandlesSubscription(**candle_data).candle[0]

    # Update trader data
    dashboard.balance = convert_balance_data(exchange.balance)
    dashboard.currency = convert_currency_data(candle)
    dashboard.profit_loss = convert_profit_loss_data(exchange.trades)

    # Render new dashboard screen
    dashboard.render()


if __name__ == "__main__":

    # find and load .env files automagically
    load_dotenv(find_dotenv())

    # Create exchange instance to initiate trade platform
    exchange = Exchange(
        getenv("BITVAVO_API_KEY"),
        getenv("BITVAVO_API_SECRET"),
    )

    # Create dashboard instance to display trade position
    dashboard = Dashboard()

    main()

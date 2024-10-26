from os import getenv

import click
from dotenv import find_dotenv, load_dotenv
from python_bitvavo_api.bitvavo import Bitvavo

from fart.data.bitvavo_data import BitvavoData


@click.command()
@click.argument("market", type=click.STRING, default="BTC-EUR")
@click.argument("interval", type=click.STRING, default="30m")
def main(market: str, interval: str) -> None:

    # Retrieve and save candle data from Bitvavo API
    bitvavo_data = BitvavoData(client)
    bitvavo_data.retrieve(market, interval)


if __name__ == "__main__":

    # Find and load .env files automagically
    load_dotenv(find_dotenv())

    # Create a Bitvavo client
    client = Bitvavo(
        {
            "APIKEY": getenv("BITVAVO_API_KEY"),
            "APISECRET": getenv("BITVAVO_API_SECRET"),
        }
    )

    # Run main function
    main()

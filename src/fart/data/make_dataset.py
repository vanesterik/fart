from os import getenv

import click
from dotenv import find_dotenv, load_dotenv

from fart.data.data_retriever import DataRetriever


@click.command()
@click.argument("market", type=click.STRING, default="BTC-EUR")
@click.argument("interval", type=click.STRING, default="30m")
def main(market: str, interval: str) -> None:

    # Retrieve and save candle data from Bitvavo API
    data_retriever.retrieve(market, interval)


if __name__ == "__main__":

    # Find and load .env files automagically
    load_dotenv(find_dotenv())

    # Create data retriever instance
    data_retriever = DataRetriever(
        getenv("BITVAVO_API_KEY"),
        getenv("BITVAVO_API_SECRET"),
    )

    # Run main function
    main()

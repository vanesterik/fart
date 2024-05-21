from os import getenv
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from loguru import logger
from python_bitvavo_api.bitvavo import Bitvavo
import click


@click.command()
@click.argument("output_dir", type=click.Path())
@click.argument("market", type=str, default="BTC-EUR")
def main(output_dir: str, market: str) -> None:
    # Get candles data from Bitvavo
    candles = client.candles(market, "1m")

    # Reverse candles list
    candles = candles[::-1]

    # Format nested list to csv
    candles = "\n".join([",".join([str(y) for y in x]) for x in candles])

    # Define output file
    filename = f"{market}.csv"
    output = (Path(output_dir) / filename).absolute()

    # Write header and candles to file
    with open(output, "w") as f:
        f.write("Timestamp,Open,High,Low,Close,Volume\n")
        f.write(candles)

    logger.info(f"Saved candles to {output}")


if __name__ == "__main__":
    # find and load .env files automagically
    load_dotenv(find_dotenv())

    # Create a Bitvavo client
    client = Bitvavo(
        {
            "APIKEY": getenv("BITVAVO_API_KEY"),
            "APISECRET": getenv("BITVAVO_API_SECRET"),
        }
    )

    main()

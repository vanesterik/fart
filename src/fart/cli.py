import sys
from os import getenv
from pathlib import Path
from typing import Annotated, Optional

import typer
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from fart.downloader import Downloader
from fart.model import train_model

app = typer.Typer(no_args_is_help=True)

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

logger.remove()
logger.add(sys.stderr, level="INFO", format=LOG_FORMAT)
logger.add("logs/cli.log", rotation="1 MB", level="INFO", format=LOG_FORMAT)

load_dotenv(find_dotenv())


@app.command()
def download(
    assets_dir: Annotated[
        str,
        typer.Option(
            help="Folder to save downloaded data (defaults to system cache directory)."
        ),
    ] = "assets",
    interval: Annotated[
        str,
        typer.Option(
            help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M')."
        ),
    ] = "1d",
    market: Annotated[
        str,
        typer.Option(help="Market to download data for (e.g., 'BTC-EUR', 'BTC-USDC')."),
    ] = "BTC-EUR",
) -> None:
    downloader = Downloader(
        api_key=getenv("BITVAVO_API_KEY"),
        api_secret=getenv("BITVAVO_API_SECRET"),
        assets_dir=Path(assets_dir),
        interval=interval,
        market=market,
    )
    downloader.download()


@app.command()
def train(
    assets_dir: Annotated[
        str,
        typer.Option(
            help="Folder to load candle data from (defaults to system cache directory)."
        ),
    ] = "assets",
    artifacts_dir: Annotated[
        str,
        typer.Option(help="Folder to save the trained model artifact to."),
    ] = "artifacts",
    interval: Annotated[
        str,
        typer.Option(
            help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M')."
        ),
    ] = "1d",
    market: Annotated[
        str,
        typer.Option(help="Market to train on (e.g., 'BTC-EUR', 'BTC-USDC')."),
    ] = "BTC-EUR",
    months: Annotated[
        Optional[int],
        typer.Option(
            help="How many months of the most recent candle history to train on. Trains on the complete cached history if not set."
        ),
    ] = None,
) -> None:
    train_model.train(
        artifacts_dir=Path(artifacts_dir),
        assets_dir=Path(assets_dir),
        interval=interval,
        market=market,
        months=months,
    )


if __name__ == "__main__":
    app()

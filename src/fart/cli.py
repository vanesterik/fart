import sys
from os import getenv
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from dotenv import find_dotenv, load_dotenv
from loguru import logger

from fart.downloader import Downloader
from fart.model import train_model


app = typer.Typer(no_args_is_help=True)

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/cli.log", rotation="1 MB", level="INFO")

load_dotenv(find_dotenv())


@app.command()
def download(
    data_dir: Annotated[
        str,
        typer.Argument(
            help="Folder to save downloaded data (defaults to system cache directory)."
        ),
    ] = "assets",
    market: Annotated[
        str,
        typer.Argument(
            help="Market to download data for (e.g., 'BTC-EUR', 'BTC-USDC')."
        ),
    ] = "BTC-EUR",
    interval: Annotated[
        str,
        typer.Argument(
            help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M')."
        ),
    ] = "1d",
) -> None:
    downloader = Downloader(
        data_dir=Path(data_dir),
        market=market,
        interval=interval,
        api_key=getenv("BITVAVO_API_KEY"),
        api_secret=getenv("BITVAVO_API_SECRET"),
    )
    downloader.download()


@app.command()
def train(
    data_dir: Annotated[
        str,
        typer.Argument(
            help="Folder to load candle data from (defaults to system cache directory)."
        ),
    ] = "assets",
    market: Annotated[
        str,
        typer.Argument(help="Market to train on (e.g., 'BTC-EUR', 'BTC-USDC')."),
    ] = "BTC-EUR",
    interval: Annotated[
        str,
        typer.Argument(
            help="Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1W', '1M')."
        ),
    ] = "1d",
    months: Annotated[
        Optional[int],
        typer.Option(
            help="How many months of the most recent candle history to train on. Trains on the complete cached history if not set."
        ),
    ] = None,
    artifacts_dir: Annotated[
        str,
        typer.Option(help="Folder to save the trained model artifact to."),
    ] = "artifacts",
    device: Annotated[
        Optional[str],
        typer.Option(
            help="Torch device to train on ('cpu', 'mps'). Auto-detected if not set."
        ),
    ] = None,
) -> None:
    train_model.train(
        data_dir=Path(data_dir),
        market=market,
        interval=interval,
        artifacts_dir=Path(artifacts_dir),
        months=months,
        device=torch.device(device) if device else None,
    )


if __name__ == "__main__":
    app()

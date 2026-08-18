import sys
from datetime import datetime, timezone
from os import getenv
from pathlib import Path
from typing import Annotated

import typer
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from tabulate import tabulate

from fart.constants import MAGNITUDE
from fart.downloader import Downloader
from fart.model.evaluate_model import evaluate_model
from fart.model.persist_model import save_model
from fart.model.prepare_datasets import prepare_datasets
from fart.model.train_model import build_model, train_model
from fart.utils import get_data_filepath, get_model_filepath

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
    num_lags: Annotated[
        int,
        typer.Option(help="Number of past candles per input window."),
    ] = 100,
    hidden_width: Annotated[
        int,
        typer.Option(help="Width of the model's hidden layers."),
    ] = 20,
    batch_size: Annotated[
        int,
        typer.Option(help="Minibatch size for training."),
    ] = 16,
    learning_rate: Annotated[
        float,
        typer.Option(help="Adam optimizer learning rate."),
    ] = 1e-3,
    num_epochs: Annotated[
        int,
        typer.Option(help="Number of training epochs."),
    ] = 500,
    train_size: Annotated[
        float,
        typer.Option(help="Proportion of windows to use for training."),
    ] = 0.8,
    n_splits: Annotated[
        int,
        typer.Option(
            help="Number of time-series CV folds. 1 disables cross-validation."
        ),
    ] = 5,
) -> None:
    data_filepath = get_data_filepath(Path(assets_dir), market, interval)

    x_train, y_train, x_test, y_test = prepare_datasets(
        data_filepath=data_filepath,
        target=MAGNITUDE,
        num_lags=num_lags,
        train_size=train_size,
    )

    model, _ = train_model(
        build_model_fn=lambda: build_model(
            num_lags=num_lags, hidden_width=hidden_width
        ),
        x_train=x_train,
        y_train=y_train,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_splits=n_splits,
    )

    (
        _,
        _,
        accuracy_train,
        accuracy_test,
        rmse_train,
        rmse_test,
        mae_train,
        mae_test,
    ) = evaluate_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    table = tabulate(
        [
            [
                "Train",
                round(accuracy_train, 2),
                round(rmse_train, 6),
                round(mae_train, 6),
            ],
            [
                "Test",
                round(accuracy_test, 2),
                round(rmse_test, 6),
                round(mae_test, 6),
            ],
        ],
        headers=["Dataset", "Accuracy", "RMSE", "MAE"],
    )
    logger.info(f"\n\n{table}\n")

    timestamp = datetime.now(timezone.utc)
    model_path = get_model_filepath(Path(artifacts_dir), market, interval, timestamp)
    save_model(model, model_path)
    logger.info(f"Saved model to {model_path}")


if __name__ == "__main__":
    app()

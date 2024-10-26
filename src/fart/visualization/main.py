import click
import matplotlib.pyplot as plt
import polars as pl

from fart.constants import feature_names as fn
from fart.features.calculate_technical_indicators import calculate_technical_indicators
from fart.features.parse_timestamp_to_datetime import parse_timestamp_to_datetime
from fart.utils.get_last_modified_data_file import get_last_modified_data_file
from fart.visualization.plot_candlestick_chart import plot_candlestick_chart


@click.command()
@click.option("--start", type=click.INT)
@click.option("--end", type=click.INT)
def main(start: int, end: int) -> None:

    last_modified_data_file = get_last_modified_data_file("./data")
    df = pl.read_csv(last_modified_data_file)
    df = parse_timestamp_to_datetime(df)
    df = calculate_technical_indicators(df)

    subset = df.tail(120).to_pandas()
    subset = subset.set_index(fn.DATETIME)

    plot_candlestick_chart(subset)


if __name__ == "__main__":

    main()

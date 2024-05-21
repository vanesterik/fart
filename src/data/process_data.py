# External imports
import datetime
import click
from loguru import logger
from pathlib import Path
import pandas as pd

# Local imports
from constants import column_names as cn


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def main(input_filepath: str, output_dir: str):
    # Read raw data
    df = pd.read_csv(input_filepath)

    # Convert Timestamp column to datetime
    df[cn.DATETIME] = pd.to_datetime(df[cn.TIMESTAMP], unit="ms")

    # Drop the old Timestamp column
    df.drop(cn.TIMESTAMP, axis=1, inplace=True)

    # Define tag for output file
    timestamp = datetime.datetime.now()
    tag = f"{timestamp.strftime('%Y%m%d-%H%M')}.csv"
    output = (Path(output_dir) / tag).absolute()

    # Write dataframe to csv
    logger.info(f"Writing file to {output}")
    df.to_csv(output, index=True)


if __name__ == "__main__":
    main()

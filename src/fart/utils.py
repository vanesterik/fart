from datetime import datetime
from pathlib import Path


# Get project root directory
def get_project_root() -> Path:
    """
    Get the root directory of the project based on specific file markers - ie.
    .git, .gitignore, or pyproject.toml.

    Returns
    -------
    - Path: Path to the root directory of the project.

    """
    # Check for common project markers
    markers = [".git", ".gitignore", "pyproject.toml"]
    current_dir = Path(__file__).parent

    while current_dir != current_dir.parent:
        if any((current_dir / marker).exists() for marker in markers):
            return current_dir
        current_dir = current_dir.parent

    # If no markers found, return the directory containing this file
    return Path(__file__).parent.parent


def get_data_filepath(data_dir: Path, market: str, interval: str) -> Path:
    """
    Get the file path for a candle data file.

    Parameters
    ----------
    - data_dir (Path): Path to the directory containing data files.
    - market (str): Market name (e.g., 'BTC-USD').
    - interval (str): Interval for the candle data (e.g., '1m', '5m', '1h').

    Returns
    -------
    - Path: Path to the candle data file.

    """
    return data_dir / f"{market}-{interval}.csv"


def get_model_filepath(
    artifacts_dir: Path, market: str, interval: str, timestamp: datetime
) -> Path:
    """
    Get the file path for a versioned model artifact. The datetime prefix
    means artifacts sort chronologically under a plain directory listing,
    and multiple training runs for the same market/interval don't
    overwrite each other.

    Parameters
    ----------
    - artifacts_dir (Path): Path to the directory to save model artifacts in.
    - market (str): Market name (e.g., 'BTC-USD').
    - interval (str): Interval for the candle data (e.g., '1m', '5m', '1h').
    - timestamp (datetime): Timestamp to prefix the file name with.

    Returns
    -------
    - Path: Path to the model artifact file.

    """
    prefix = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    return artifacts_dir / f"{prefix}-{market}-{interval}.pt"


def get_latest_model_filepath(artifacts_dir: Path, market: str, interval: str) -> Path:
    """
    Get the most recently trained model artifact for a market and interval,
    determined by the artifact file name's datetime prefix (not filesystem
    modification time, which copies/checkouts can alter).

    Parameters
    ----------
    - artifacts_dir (Path): Path to the directory model artifacts are saved in.
    - market (str): Market name (e.g., 'BTC-USD').
    - interval (str): Interval for the candle data (e.g., '1m', '5m', '1h').

    Returns
    -------
    - Path: Path to the most recent model artifact file.

    """
    file_list = list(artifacts_dir.glob(f"*-{market}-{interval}.pt"))

    return max(file_list, key=lambda f: f.name)


def get_last_modified_data_file(data_dir: str) -> Path:
    """
    Get the last modified data file in the given directory.

    Parameters
    ----------
    - data_dir (Path): Path to the directory containing data files.

    Returns
    -------
    - Path: Path to the last modified data file.

    """

    # Get file list of csv data files in passed data directory
    file_list = list(Path(data_dir).glob("*.csv"))

    # Determine and return last modified data file of file list
    return max(file_list, key=lambda f: f.stat().st_mtime)

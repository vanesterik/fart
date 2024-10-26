from pathlib import Path


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

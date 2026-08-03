import os
import tempfile
import time
from pathlib import Path

import pytest

from fart.utils.get_last_modified_data_file import get_last_modified_data_file


def test_get_last_modified_data_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:

        file1 = Path(temp_dir) / "file1.csv"
        file2 = Path(temp_dir) / "file2.csv"
        file3 = Path(temp_dir) / "file3.csv"

        file1.touch()
        file2.touch()
        file3.touch()

        assert get_last_modified_data_file(temp_dir) == file3


def test_get_last_modified_data_file_empty_dir() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError):
            get_last_modified_data_file(temp_dir)


def test_get_last_modified_data_file_no_csv_files() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:

        file1 = Path(temp_dir) / "file1.txt"
        file2 = Path(temp_dir) / "file2.doc"

        file1.touch()
        file2.touch()

        with pytest.raises(ValueError):
            get_last_modified_data_file(temp_dir)

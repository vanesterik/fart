def print_in_columns(left_col: str, right_col: str, column_width: int = 16) -> None:
    """
    Print two strings in columns with a specified width.

    Parameters
    ----------
    - left_col: str
    - right_col: str
    - column_width: int

    Returns
    -------
    - None

    Example
    -------
    >>> print_in_columns("Train Score:", "90.00%", 16)
    Train Score:     90.00%

    """
    print(f"{left_col:<{column_width}}{right_col:>{column_width}}")

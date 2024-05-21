from .print_in_columns import print_in_columns


def print_scores(train_score: float, test_score: float) -> None:
    """
    Print the train and test scores of a model.

    Parameters
    ----------
    - train_score: float
    - test_score: float

    Returns
    -------
    - None

    Example
    -------
    >>> print_scores(0.9, 0.8)
    Train score: 90.00%
    Test score: 80.00%

    """
    print_in_columns("Train score:", f"{train_score * 100:.2f}%", 12)
    print_in_columns("Test score:", f"{test_score * 100:.2f}%", 12)

import polars as pl

from fart.constants import feature_names as fn
from fart.model.train_test_split import train_test_split


def test_train_test_split() -> None:
    # Create a sample DataFrame
    data = {
        fn.TIMESTAMP: range(10),
        fn.TRADE_SIGNAL: [0, 0, 0, 1, 0, 0, -1, 0, 0, 0],
    }
    df = pl.DataFrame(data)

    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(df)

    # Check the shapes of the splits
    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_test.shape[0] == 2

    # Check the content of the splits
    assert X_train.iloc[0][fn.TIMESTAMP] == 0
    assert X_test.iloc[0][fn.TIMESTAMP] == 8
    assert y_train.iloc[0] == 0
    assert y_test.iloc[0] == 0

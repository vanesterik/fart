from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from fart.common import constants as cl


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int] = [cl.HOLD_CLASS, cl.BUY_CLASS, cl.SELL_CLASS],
) -> None:
    """
    Plot confusion matrix.

    Parameters
    ----------
    - y_true (List[int]): True labels.
    - y_pred (List[int]): Predicted labels.
    - labels (List[int], optional): List of labels. Defaults to [cl.HOLD,
      cl.BUY, cl.SELL].

    """

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.subplots(figsize=(4, 3))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="viridis", xticklabels=labels, yticklabels=labels
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.show()

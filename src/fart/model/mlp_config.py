from pydantic import BaseModel


class MLPConfig(BaseModel):
    """
    Configuration for the MLP regressor.

    Attributes
    ----------
    - in_features (int): Number of input features (the lag window
      width passed as `num_lags` to `prepare_datasets`).
    - hidden_features (int): Width of each hidden layer.
    - out_features (int): Number of output features. Defaults to 1;
      `train_model`/`evaluate_model` currently assume 1.
    - num_layers (int): Number of hidden layers.
    - dropout_rate (float): Dropout probability applied in every block.
      Defaults to 0.2.

    """

    in_features: int
    hidden_features: int
    out_features: int = 1
    num_layers: int
    dropout_rate: float = 0.2

from pydantic import BaseModel


class CNNConfig(BaseModel):
    """
    Configuration for the CNN regressor.

    Attributes
    ----------
    - in_features (int): Number of input features (the lag window
      width passed as `num_lags` to `prepare_datasets`).
    - hidden_channels (int): Number of output channels for every block
      (fixed width, matching `MLPConfig.hidden_features`'s convention).
    - out_features (int): Number of output features. Defaults to 1.
    - num_layers (int): Number of Conv1d->BatchNorm1d->ReLU->Dropout
      blocks.
    - kernel_size (int): Convolution kernel width, used by every block.
    - dropout_rate (float): Dropout probability applied in every block.
      Defaults to 0.2, matching `MLPConfig`.

    """

    in_features: int
    hidden_channels: int
    out_features: int = 1
    num_layers: int
    kernel_size: int
    dropout_rate: float = 0.2

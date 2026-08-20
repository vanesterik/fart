from pydantic import BaseModel


class CNNConfig(BaseModel):
    """
    Configuration for the CNN regressor.

    Attributes
    ----------
    - num_lags (int): Width of the input lag window (must match the
      `num_lags` passed to `prepare_datasets`).
    - num_blocks (int): Number of Conv1d->BatchNorm1d->ReLU->Dropout
      blocks.
    - num_channels (int): Number of output channels for every block
      (fixed width, matching `MLPConfig.num_neurons`'s convention).
    - kernel_size (int): Convolution kernel width, used by every block.
    - dropout (float): Dropout probability applied in every block.
      Defaults to 0.2, matching `MLPConfig`.

    """

    num_lags: int
    num_blocks: int
    num_channels: int
    kernel_size: int
    dropout: float = 0.2

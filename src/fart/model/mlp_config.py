from pydantic import BaseModel


class MLPConfig(BaseModel):
    """
    Configuration for the MLP regressor.

    Attributes
    ----------
    - num_lags (int): Width of the input lag window (must match the
      `num_lags` passed to `prepare_datasets`).
    - num_blocks (int): Number of Linear->BatchNorm1d->ReLU->Dropout
      blocks.
    - num_neurons (int): Width of each block's Linear/BatchNorm1d layers.
    - dropout (float): Dropout probability applied in every block.
      Defaults to 0.2, matching the value previously hardcoded in
      `build_mlp_model`.

    """

    num_lags: int
    num_blocks: int
    num_neurons: int
    dropout: float = 0.2

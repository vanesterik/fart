from typing import Protocol

from torch import nn


class ModelBuilder(Protocol):
    """
    Structural interface for a builder that assembles a fresh, untrained
    `nn.Module` from its own config.

    Any object exposing a zero-arg `build() -> nn.Module` satisfies this
    Protocol -- no inheritance required. A builder's `build()` call
    produces the `model` passed to `train_model`, so adding a builder
    never requires changing `train_model.py`.

    Future architecture sketch (not implemented here -- CNN/GRU/N-BEATS/
    transformer are separate, not-yet-started PRD stories):

        class CNNConfig(BaseModel):
            num_lags: int
            num_channels: int
            kernel_size: int = 3

        class CNNBuilder:
            def __init__(self, config: CNNConfig) -> None:
                self._config = config

            def build(self) -> nn.Module:
                ...  # conv_block(...) from blocks.py, etc.

    Swapping MLP for CNN at a call site then becomes
    `model=CNNBuilder(cnn_config).build()` -- a config/builder choice, not
    a rewrite of `train_model` or the notebook's training call.

    """

    def build(self) -> nn.Module: ...

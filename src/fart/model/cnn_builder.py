from torch import nn

from fart.model.blocks import conv_block
from fart.model.cnn_config import CNNConfig


class CNNBuilder:
    """
    Builds a CNN regressor. GoF Builder pattern: construction
    (`CNNConfig`) is decoupled from assembly (`build`), so swapping
    architectures at a `train_model` call site is a matter of choosing a
    different `ModelBuilder`, not rewriting a model-construction
    function.

    Attributes
    ----------
    - config (CNNConfig): Architecture hyperparameters.

    """

    def __init__(self, config: CNNConfig) -> None:
        self._config = config

    def build(self) -> nn.Module:
        """
        Assemble a fresh, untrained CNN: an `Unflatten` to add the
        channel dimension `Conv1d` expects, `config.num_blocks` repeats
        of `conv_block` (each preserving sequence length via
        `padding="same"`), global average pooling down to one value per
        channel, then a final `Linear` to a scalar output.

        Global pooling means the output shape never depends on
        `config.num_lags`/`config.kernel_size`/`config.num_blocks`
        combining awkwardly -- no manual output-length arithmetic.

        Each call constructs new `nn.Conv1d`/`nn.BatchNorm1d`/`nn.Linear`
        layers (and therefore freshly initialized weights) -- calling
        `build()` twice returns two independent models sharing no state.

        Returns
        -------
        - nn.Module: An untrained `nn.Sequential` CNN.

        """
        layers: list[nn.Module] = [nn.Unflatten(1, (1, self._config.num_lags))]
        in_channels = 1

        for _ in range(self._config.num_blocks):
            layers.append(
                conv_block(
                    in_channels=in_channels,
                    out_channels=self._config.num_channels,
                    kernel_size=self._config.kernel_size,
                    dropout=self._config.dropout,
                )
            )
            in_channels = self._config.num_channels

        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(in_channels, 1))

        return nn.Sequential(*layers)

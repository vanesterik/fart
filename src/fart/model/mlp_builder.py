from torch import nn

from fart.model.blocks import linear_block
from fart.model.mlp_config import MLPConfig


class MLPBuilder:
    """
    Builds an MLP regressor. GoF Builder pattern: construction
    (`MLPConfig`) is decoupled from assembly (`build`), so swapping
    architectures at a `train_model` call site is a matter of choosing a
    different `ModelBuilder`, not rewriting a model-construction
    function.

    Attributes
    ----------
    - config (MLPConfig): Architecture hyperparameters.

    """

    def __init__(self, config: MLPConfig) -> None:
        self._config = config

    def build(self) -> nn.Module:
        """
        Assemble a fresh, untrained MLP: `config.num_layers` repeats of
        `linear_block`, narrowing/widening from `config.in_features` to
        `config.hidden_features`, followed by a final `Linear` to
        `config.out_features` outputs.

        Each call constructs new `nn.Linear`/`nn.BatchNorm1d` layers (and
        therefore freshly initialized weights) -- calling `build()` twice
        returns two independent models sharing no state.

        Returns
        -------
        - nn.Module: An untrained `nn.Sequential` MLP.

        """
        layers: list[nn.Module] = []
        prev_features = self._config.in_features

        for _ in range(self._config.num_layers):
            layers.append(
                linear_block(
                    in_features=prev_features,
                    out_features=self._config.hidden_features,
                    dropout_rate=self._config.dropout_rate,
                )
            )
            prev_features = self._config.hidden_features

        layers.append(nn.Linear(prev_features, self._config.out_features))

        return nn.Sequential(*layers)

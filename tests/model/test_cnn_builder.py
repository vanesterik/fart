import torch
from torch import nn

from fart.model.builder import ModelBuilder
from fart.model.cnn_builder import CNNBuilder
from fart.model.cnn_config import CNNConfig


def test_cnn_builder_layer_count_and_output_shape() -> None:
    config = CNNConfig(num_lags=10, num_blocks=2, num_channels=4, kernel_size=3)
    model = CNNBuilder(config).build()
    model.eval()

    assert isinstance(model, nn.Sequential)
    # Unflatten + 2 conv blocks + AdaptiveAvgPool1d + Flatten + final Linear
    assert len(model) == 1 + 2 + 1 + 1 + 1

    output = model(torch.zeros(1, 10))
    assert output.shape == (1, 1)


def test_cnn_config_dropout_default() -> None:
    config = CNNConfig(num_lags=5, num_blocks=1, num_channels=4, kernel_size=3)

    assert config.dropout == 0.2


def test_cnn_builder_dropout_override() -> None:
    config = CNNConfig(
        num_lags=5, num_blocks=1, num_channels=4, kernel_size=3, dropout=0.5
    )
    model = CNNBuilder(config).build()

    dropout_layers = [
        layer for layer in model.modules() if isinstance(layer, nn.Dropout)
    ]
    assert dropout_layers
    assert all(layer.p == 0.5 for layer in dropout_layers)


def test_cnn_builder_build_is_independent_per_call() -> None:
    config = CNNConfig(num_lags=5, num_blocks=1, num_channels=4, kernel_size=3)
    builder = CNNBuilder(config)

    first = builder.build()
    second = builder.build()

    assert first is not second

    first_conv = next(
        layer for layer in first.modules() if isinstance(layer, nn.Conv1d)
    )
    second_conv = next(
        layer for layer in second.modules() if isinstance(layer, nn.Conv1d)
    )
    with torch.no_grad():
        first_conv.weight.fill_(0.0)
    assert not torch.equal(first_conv.weight, second_conv.weight)


def test_cnn_builder_satisfies_model_builder_protocol() -> None:
    config = CNNConfig(num_lags=5, num_blocks=1, num_channels=4, kernel_size=3)
    builder: ModelBuilder = CNNBuilder(config)

    assert isinstance(builder.build(), nn.Module)


def test_cnn_builder_output_shape_independent_of_num_lags() -> None:
    config = CNNConfig(num_lags=100, num_blocks=3, num_channels=8, kernel_size=5)
    model = CNNBuilder(config).build()
    model.eval()

    output = model(torch.zeros(2, 100))
    assert output.shape == (2, 1)

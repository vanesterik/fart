import torch
from torch import nn

from fartt.model.builder import ModelBuilder
from fartt.model.mlp_builder import MLPBuilder
from fartt.model.mlp_config import MLPConfig


def test_mlp_builder_layer_count_and_output_shape() -> None:
    config = MLPConfig(in_features=10, num_layers=2, hidden_features=4)
    model = MLPBuilder(config).build()
    model.eval()

    assert isinstance(model, nn.Sequential)
    assert len(model) == 2 + 1  # 2 blocks + final Linear

    output = model(torch.zeros(1, 10))
    assert output.shape == (1, 1)


def test_mlp_config_dropout_default() -> None:
    config = MLPConfig(in_features=5, num_layers=1, hidden_features=3)

    assert config.dropout_rate == 0.2


def test_mlp_builder_out_features() -> None:
    config = MLPConfig(in_features=5, hidden_features=3, num_layers=1, out_features=2)
    model = MLPBuilder(config).build()
    model.eval()

    output = model(torch.zeros(1, 5))
    assert output.shape == (1, 2)


def test_mlp_builder_dropout_override() -> None:
    config = MLPConfig(in_features=5, num_layers=1, hidden_features=3, dropout_rate=0.5)
    model = MLPBuilder(config).build()

    dropout_layers = [
        layer for layer in model.modules() if isinstance(layer, nn.Dropout)
    ]
    assert dropout_layers
    assert all(layer.p == 0.5 for layer in dropout_layers)


def test_mlp_builder_build_is_independent_per_call() -> None:
    config = MLPConfig(in_features=5, num_layers=1, hidden_features=3)
    builder = MLPBuilder(config)

    first = builder.build()
    second = builder.build()

    assert first is not second

    first_linear = next(
        layer for layer in first.modules() if isinstance(layer, nn.Linear)
    )
    second_linear = next(
        layer for layer in second.modules() if isinstance(layer, nn.Linear)
    )
    with torch.no_grad():
        first_linear.weight.fill_(0.0)
    assert not torch.equal(first_linear.weight, second_linear.weight)


def test_mlp_builder_satisfies_model_builder_protocol() -> None:
    config = MLPConfig(in_features=5, num_layers=1, hidden_features=3)
    builder: ModelBuilder = MLPBuilder(config)

    assert isinstance(builder.build(), nn.Module)

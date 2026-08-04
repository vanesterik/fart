from fart.model.nbeats_config import NBeatsConfig


def test_nbeats_config_defaults() -> None:
    config = NBeatsConfig()

    assert config.lookback == 30
    assert config.num_stacks == 2
    assert config.num_blocks_per_stack == 3
    assert config.hidden_width == 64
    assert config.epochs == 50
    assert config.learning_rate == 1e-3


def test_nbeats_config_overrides() -> None:
    config = NBeatsConfig(epochs=2, num_stacks=1, num_blocks_per_stack=1)

    assert config.epochs == 2
    assert config.num_stacks == 1
    assert config.num_blocks_per_stack == 1

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


def test_nbeats_config_default_batch_size() -> None:
    config = NBeatsConfig()

    assert config.batch_size == 128


def test_nbeats_config_batch_size_override() -> None:
    config = NBeatsConfig(batch_size=4)

    assert config.batch_size == 4


def test_nbeats_config_default_beta_nll() -> None:
    config = NBeatsConfig()

    assert config.beta_nll == 0.0


def test_nbeats_config_beta_nll_override() -> None:
    config = NBeatsConfig(beta_nll=1.0)

    assert config.beta_nll == 1.0

from pathlib import Path

import torch

from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig
from fart.model.nbeats_persistence import load_model, save_model


def test_save_and_load_model_round_trip(tmp_path: Path) -> None:
    config = NBeatsConfig(
        lookback=10, num_stacks=1, num_blocks_per_stack=1, hidden_width=8
    )
    model = NBeatsNet(config)
    model.eval()

    path = tmp_path / "checkpoint.pt"
    save_model(model, config, path)
    loaded_model, loaded_config = load_model(path)

    assert loaded_model.state_dict().keys() == model.state_dict().keys()

    x = torch.randn(3, loaded_config.lookback)
    with torch.no_grad():
        original_output = model(x)
        loaded_output = loaded_model(x)

    assert torch.allclose(original_output, loaded_output)


def test_save_model_creates_missing_parent_directory(tmp_path: Path) -> None:
    config = NBeatsConfig(
        lookback=10, num_stacks=1, num_blocks_per_stack=1, hidden_width=8
    )
    model = NBeatsNet(config)
    path = tmp_path / "nested" / "checkpoint.pt"

    save_model(model, config, path)

    assert path.exists()

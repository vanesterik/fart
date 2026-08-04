import torch

from fart.model.nbeats import NBeatsNet
from fart.model.nbeats_config import NBeatsConfig


def test_nbeats_net_forward_output_shape() -> None:
    config = NBeatsConfig(
        lookback=10, num_stacks=1, num_blocks_per_stack=2, hidden_width=8
    )
    model = NBeatsNet(config)
    x = torch.randn(4, config.lookback)

    output = model(x)

    assert output.shape == (4, 2)


def test_nbeats_net_training_step_produces_finite_gradients() -> None:
    config = NBeatsConfig(
        lookback=10, num_stacks=1, num_blocks_per_stack=2, hidden_width=8
    )
    model = NBeatsNet(config)
    x = torch.randn(4, config.lookback)
    target = torch.randn(4)

    output = model(x)
    mu, log_sigma = output.unbind(-1)
    loss_fn = torch.nn.GaussianNLLLoss()
    loss = loss_fn(mu, target, log_sigma.exp() ** 2)
    loss.backward()

    # The last block's backcast residual is never consumed (no further block
    # reads it), so its backcast_layer weights are structurally outside the
    # autograd graph — that's the "backcast is architectural only" property,
    # not a bug. Every other parameter must still get a finite gradient.
    last_block_backcast_prefix = f"blocks.{len(model.blocks) - 1}.backcast_layer"
    for name, param in model.named_parameters():
        if name.startswith(last_block_backcast_prefix):
            continue
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name

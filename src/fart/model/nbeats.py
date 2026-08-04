from typing import Tuple

import torch
from torch import Tensor, nn

from fart.model.nbeats_config import NBeatsConfig

FORECAST_WIDTH = 2


class NBeatsBlock(nn.Module):
    """One generic N-BEATS block: FC stack -> backcast + forecast(width=2)."""

    def __init__(self, lookback: int, hidden_width: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.fc = nn.Sequential(
            nn.Linear(lookback, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
        )
        self.backcast_layer = nn.Linear(hidden_width, lookback)
        self.forecast_layer = nn.Linear(hidden_width, FORECAST_WIDTH)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        hidden = self.fc(x)
        backcast = self.backcast_layer(hidden)
        forecast = self.forecast_layer(hidden)
        return backcast, forecast


class NBeatsNet(nn.Module):
    """Stack of NBeatsBlocks, doubly-residual, generic (non-interpretable) basis."""

    def __init__(self, config: NBeatsConfig) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        num_blocks = config.num_stacks * config.num_blocks_per_stack
        self.blocks = nn.ModuleList(
            [
                NBeatsBlock(config.lookback, config.hidden_width)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        forecast = torch.zeros(
            x.shape[0], FORECAST_WIDTH, dtype=x.dtype, device=x.device
        )
        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        return forecast

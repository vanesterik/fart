from torch import nn


def build_mlp_model(num_lags: int, num_blocks: int, num_neurons: int) -> nn.Module:
    layers: list[nn.Module] = []
    prev_dim = num_lags

    for _ in range(num_blocks):
        layers.append(nn.Linear(prev_dim, num_neurons))
        layers.append(nn.BatchNorm1d(num_neurons))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=0.2))
        prev_dim = num_neurons

    layers.append(nn.Linear(prev_dim, 1))

    return nn.Sequential(*layers)

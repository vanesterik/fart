import torch
from torch import nn

from fart.model.blocks import conv_block, linear_block


def test_linear_block_layer_types_and_dims() -> None:
    block = linear_block(in_features=3, out_features=5, dropout=0.2)

    children = list(block.children())
    assert [type(layer) for layer in children] == [
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Dropout,
    ]

    linear, batch_norm, _, dropout = children
    assert linear.in_features == 3
    assert linear.out_features == 5
    assert batch_norm.num_features == 5
    assert dropout.p == 0.2


def test_linear_block_output_shape() -> None:
    block = linear_block(in_features=3, out_features=5, dropout=0.2)
    block.eval()

    output = block(torch.zeros(1, 3))

    assert output.shape == (1, 5)


def test_conv_block_layer_types_and_dims() -> None:
    block = conv_block(in_channels=1, out_channels=4, kernel_size=3, dropout=0.2)

    children = list(block.children())
    assert [type(layer) for layer in children] == [
        nn.Conv1d,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Dropout,
    ]

    conv, batch_norm, _, dropout = children
    assert conv.in_channels == 1
    assert conv.out_channels == 4
    assert conv.kernel_size == (3,)
    assert batch_norm.num_features == 4
    assert dropout.p == 0.2


def test_conv_block_output_shape_preserves_length() -> None:
    block = conv_block(in_channels=1, out_channels=4, kernel_size=3, dropout=0.2)
    block.eval()

    output = block(torch.zeros(1, 1, 10))

    assert output.shape == (1, 4, 10)

from typing import Protocol

from torch import nn


class ModelBuilder(Protocol):
    """
    Structural interface for a builder that assembles a fresh, untrained
    `nn.Module` from its own config.

    Any object exposing a zero-arg `build() -> nn.Module` satisfies this
    Protocol -- no inheritance required. A builder's `build()` call
    produces the `model` passed to `train_model`, so adding a builder
    never requires changing `train_model.py`.

    `MLPBuilder` (`mlp_builder.py`) and `CNNBuilder` (`cnn_builder.py`)
    are the implementations so far; each takes its own typed config.
    Swapping MLP for CNN at a call site is
    `model=CNNBuilder(cnn_config).build()` -- a config/builder choice, not
    a rewrite of `train_model` or the notebook's training call.

    """

    def build(self) -> nn.Module: ...

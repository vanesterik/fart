from pydantic import BaseModel


class NBeatsConfig(BaseModel):
    """
    Configuration for the quick-prototype N-BEATS model.

    Attributes
    ----------
    - lookback (int): Backcast length, in candles.
    - num_stacks (int): Number of stacks of blocks.
    - num_blocks_per_stack (int): Number of blocks per stack.
    - hidden_width (int): Width of each block's hidden fully-connected layers.
    - epochs (int): Number of training epochs.
    - learning_rate (float): Adam optimizer learning rate.

    """

    lookback: int = 30
    num_stacks: int = 2
    num_blocks_per_stack: int = 3
    hidden_width: int = 64
    epochs: int = 50
    learning_rate: float = 1e-3

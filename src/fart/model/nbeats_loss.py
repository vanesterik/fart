import torch.nn.functional as F
from torch import Tensor


def beta_nll_loss(mu: Tensor, target: Tensor, log_sigma: Tensor, beta: float) -> Tensor:
    """
    Beta-NLL loss (Seitzer et al., ICLR 2022): weights each sample's
    Gaussian negative log-likelihood by its own predicted variance,
    stop-gradient, raised to `beta`. Plain Gaussian NLL lets a model lower
    its loss by uniformly shrinking predicted variance instead of learning
    which windows are genuinely harder to predict -- beta-NLL removes that
    incentive by scaling down the loss contribution of already-confident
    (low-variance) predictions.

    Parameters
    ----------
    - mu (Tensor): Predicted mean, shape (batch,).
    - target (Tensor): Ground-truth value, shape (batch,).
    - log_sigma (Tensor): Predicted log standard deviation, shape (batch,).
    - beta (float): Variance-weighting exponent. `0.0` recovers plain
      Gaussian NLL; `1.0` approximates MSE-style weighting on the mean.

    Returns
    -------
    - Tensor: Scalar loss, averaged over the batch.

    """
    var = log_sigma.exp() ** 2
    nll = F.gaussian_nll_loss(mu, target, var, reduction="none")
    weight = var.detach() ** beta
    return (weight * nll).mean()

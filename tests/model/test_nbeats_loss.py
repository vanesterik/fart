import math

import torch
from torch import nn

from fart.model.nbeats_loss import beta_nll_loss


def test_beta_nll_loss_at_beta_zero_matches_gaussian_nll_loss() -> None:
    mu = torch.tensor([0.1, -0.4, 0.7])
    target = torch.tensor([0.2, -0.1, 0.5])
    log_sigma = torch.tensor([-1.0, 0.3, 0.8])
    var = log_sigma.exp() ** 2

    actual = beta_nll_loss(mu, target, log_sigma, beta=0.0)
    expected = nn.GaussianNLLLoss()(mu, target, var)

    torch.testing.assert_close(actual, expected)


def test_beta_nll_loss_matches_hand_computed_weighted_example() -> None:
    mu = torch.tensor([0.0, 0.0])
    target = torch.tensor([1.0, 2.0])
    log_sigma = torch.tensor([0.0, 1.0])
    beta = 0.5

    var = log_sigma.exp() ** 2
    nll = 0.5 * (torch.log(var) + (target - mu) ** 2 / var)
    weight = var**beta
    expected = (weight * nll).mean()

    actual = beta_nll_loss(mu, target, log_sigma, beta=beta)

    torch.testing.assert_close(actual, expected)
    assert math.isclose(expected.item(), 1.977020, rel_tol=1e-5)


def test_beta_nll_loss_gradients_are_finite() -> None:
    mu = torch.tensor([0.1, -0.4, 0.7], requires_grad=True)
    target = torch.tensor([0.2, -0.1, 0.5])
    log_sigma = torch.tensor([-1.0, 0.3, 0.8], requires_grad=True)

    loss = beta_nll_loss(mu, target, log_sigma, beta=0.5)
    loss.backward()

    assert mu.grad is not None
    assert torch.isfinite(mu.grad).all()
    assert log_sigma.grad is not None
    assert torch.isfinite(log_sigma.grad).all()

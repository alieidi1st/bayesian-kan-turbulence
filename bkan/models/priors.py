"""
Prior distributions and helpers for variational Bayesian neural networks.

Provides the three prior families used across the BNN and BKAN models, each
taken from the original literature:

* Gaussian prior with closed-form KL          -- Graves (2011)
* Scale-mixture-of-Gaussians prior, MC KL     -- Blundell et al. (2015)
* Gamma hyperprior on the precision (ARD)     -- MacKay (1992); Bishop PRML s10

All functions operate on flat tensors of (sampled or distributional) weights and
return scalar contributions, so they can be reused by both the linear (MLP) and
spline (KAN) variational layers.
"""

import math

import torch
from torch import Tensor

_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_log_prob(w: Tensor, sigma: float) -> Tensor:
    """Sum of log N(w_i; 0, sigma^2) over all elements of w."""
    var = sigma * sigma
    return torch.sum(
        -0.5 * _LOG_2PI - math.log(sigma) - w.pow(2) / (2.0 * var)
    )


def scale_mixture_log_prob(
    w: Tensor, pi: float, sigma1: float, sigma2: float
) -> Tensor:
    """
    Sum of log p(w_i) under the Blundell (2015) scale-mixture prior

        p(w) = pi * N(0, sigma1^2) + (1 - pi) * N(0, sigma2^2),

    computed elementwise with a numerically stable log-sum-exp.
    """
    var1, var2 = sigma1 * sigma1, sigma2 * sigma2
    log_n1 = -0.5 * _LOG_2PI - math.log(sigma1) - w.pow(2) / (2.0 * var1)
    log_n2 = -0.5 * _LOG_2PI - math.log(sigma2) - w.pow(2) / (2.0 * var2)
    # log( pi*N1 + (1-pi)*N2 ) elementwise
    log_mix = torch.logaddexp(
        math.log(pi) + log_n1,
        math.log(1.0 - pi) + log_n2,
    )
    return torch.sum(log_mix)


def gaussian_posterior_log_prob(w: Tensor, mu: Tensor, sigma: Tensor) -> Tensor:
    """Sum of log N(w_i; mu_i, sigma_i^2): variational posterior density at w."""
    return torch.sum(
        -0.5 * _LOG_2PI - torch.log(sigma) - (w - mu).pow(2) / (2.0 * sigma.pow(2))
    )


def gaussian_kl(mu: Tensor, var: Tensor, prior_var: float) -> Tensor:
    """
    Closed-form KL( N(mu, var) || N(0, prior_var) ), summed over elements.

    KL = 0.5 * sum[ var/prior_var + mu^2/prior_var - 1 - log var + log prior_var ].
    """
    return 0.5 * torch.sum(
        var / prior_var
        + mu.pow(2) / prior_var
        - 1.0
        - torch.log(var)
        + math.log(prior_var)
    )


def gamma_adaptive_prior_variance(
    sq_sum: Tensor, n_params: int, a0: float = 1e-3, b0: float = 1e-3
) -> Tensor:
    """
    Empirical-Bayes prior variance under a Gamma hyperprior on the precision.

    With a hierarchical prior w ~ N(0, 1/lambda), lambda ~ Gamma(a0, b0), the
    mean of the conjugate Gamma posterior over the precision given the current
    weight statistics is

        lambda* = (a0 + n/2) / (b0 + 0.5 * E_q[sum w^2]),

    and the prior variance used in the (Gaussian) KL term is 1 / lambda*. This
    is the ARD / evidence-framework update (MacKay 1992; Bishop PRML s10.1).
    The statistic sq_sum = E_q[sum w^2] = sum(mu^2 + var) should be passed
    detached so that lambda* is treated as a constant in the weight gradient
    step (variational EM).

    Returns
    -------
    prior_var : Tensor (scalar)
    """
    lam = (a0 + 0.5 * n_params) / (b0 + 0.5 * sq_sum)
    return 1.0 / lam

"""
Standard deterministic and ensemble baselines for uncertainty quantification.

Implements the canonical non-variational baselines used throughout the BNN /
UQ literature, for fair comparison against BayesianKAN and BayesianNN:

* DeterministicMLP : plain feedforward network with a Gaussian negative
  log-likelihood head (a single Deep-Ensemble member). Serves as the point
  prediction / accuracy baseline.

* DeepEnsemble : ensemble of independently initialised DeterministicMLP
  networks (Lakshminarayanan et al. 2017, NeurIPS). The predictive
  distribution is the Gaussian mixture of the members, with the standard
  epistemic/aleatoric decomposition.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class DeterministicMLP(nn.Module):
    """
    Feedforward MLP with an optional heteroscedastic Gaussian likelihood head.

    When learn_noise is True the network outputs both a mean and a
    log-variance, and is trained on the Gaussian negative log-likelihood
    (a single Deep-Ensemble member, Lakshminarayanan et al. 2017). When
    learn_noise is False it outputs only a mean and is trained on MSE.

    Parameters
    ----------
    input_dim : int
        Input feature dimension. Default: 1.
    hidden_dims : list of int
        Hidden layer widths. Default: [50, 50].
    output_dim : int
        Output dimension. Default: 1.
    learn_noise : bool
        If True, predict heteroscedastic log-variance and use Gaussian NLL.
        Default: True.
    activation : str
        One of 'relu', 'tanh', 'elu'. Default: 'tanh'.
    log_var_min, log_var_max : float
        Clamping bounds on the predicted log-variance for numerical stability.
        Defaults: -20.0, 5.0.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        learn_noise: bool = True,
        activation: str = 'tanh',
        log_var_min: float = -20.0,
        log_var_max: float = 5.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [50, 50]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learn_noise = learn_noise
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'elu': nn.ELU()}.get(
            activation, nn.Tanh())

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), act]
            prev = h
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(prev, output_dim)
        if learn_noise:
            self.log_var_head = nn.Linear(prev, output_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Return (mean, log_var). log_var is None if learn_noise is False."""
        h = self.backbone(x)
        mean = self.mean_head(h)
        if self.learn_noise:
            log_var = torch.clamp(
                self.log_var_head(h), self.log_var_min, self.log_var_max)
            return mean, log_var
        return mean, None

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Gaussian NLL (heteroscedastic) or MSE."""
        mean, log_var = self.forward(x)
        if self.learn_noise:
            var = torch.exp(log_var) + 1e-6
            return 0.5 * torch.mean(log_var + (y - mean) ** 2 / var)
        return 0.5 * torch.mean((y - mean) ** 2)

    @torch.no_grad()
    def predict(self, x: Tensor, n_samples: int = None
                ) -> Tuple[Tensor, Tensor, None]:
        """
        Mean and total std. A single deterministic network has no epistemic
        uncertainty, so the total std is the predicted aleatoric std (or zero
        if learn_noise is False).
        """
        self.eval()
        mean, log_var = self.forward(x)
        if log_var is not None:
            std = torch.sqrt(torch.exp(log_var) + 1e-6)
        else:
            std = torch.zeros_like(mean)
        return mean, std, None

    @torch.no_grad()
    def predict_decomposed(self, x: Tensor, n_samples: int = None
                           ) -> Tuple[Tensor, Tensor, Tensor]:
        self.eval()
        mean, log_var = self.forward(x)
        epistemic_std = torch.zeros_like(mean)
        aleatoric_std = (torch.sqrt(torch.exp(log_var) + 1e-6)
                         if log_var is not None else torch.zeros_like(mean))
        return mean, epistemic_std, aleatoric_std


class DeepEnsemble(nn.Module):
    """
    Deep Ensemble for uncertainty quantification (Lakshminarayanan et al. 2017).

    An ensemble of `n_members` DeterministicMLP networks, each with a
    heteroscedastic Gaussian head and independent random initialisation.
    The predictive distribution is a uniformly weighted Gaussian mixture; its
    mean and variance follow the standard mixture moments:

        mu*(x)      = (1/M) sum_m  mu_m(x)
        var*(x)     = (1/M) sum_m [ sigma_m^2(x) + mu_m^2(x) ]  -  mu*(x)^2
                    = aleatoric  +  epistemic
        aleatoric   = (1/M) sum_m  sigma_m^2(x)        (mean of variances)
        epistemic   = (1/M) sum_m  mu_m^2(x) - mu*^2   (variance of means)

    Parameters
    ----------
    n_members : int
        Number of ensemble members. Default: 5 (the value used in the paper).
    input_dim, hidden_dims, output_dim, activation :
        Passed to each DeterministicMLP member.
    learn_noise : bool
        If True, members use heteroscedastic Gaussian NLL. Default: True.
    """

    def __init__(
        self,
        n_members: int = 5,
        input_dim: int = 1,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        learn_noise: bool = True,
        activation: str = 'tanh',
    ):
        super().__init__()
        self.n_members = n_members
        self.output_dim = output_dim
        self.learn_noise = learn_noise
        self.members = nn.ModuleList([
            DeterministicMLP(
                input_dim=input_dim, hidden_dims=hidden_dims,
                output_dim=output_dim, learn_noise=learn_noise,
                activation=activation,
            )
            for _ in range(n_members)
        ])

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Mean prediction of member 0 (single-member forward; use predict())."""
        return self.members[0](x)

    @torch.no_grad()
    def predict(
        self, x: Tensor, n_samples: int = None
    ) -> Tuple[Tensor, Tensor, None]:
        """
        Predictive mean and total standard deviation of the Gaussian mixture.

        The n_samples argument is accepted for interface compatibility with
        the variational models but is ignored (the ensemble size is fixed).

        Returns
        -------
        mean : Tensor, shape (batch, output_dim)
        std : Tensor, shape (batch, output_dim)
        None
        """
        self.eval()
        means, variances = [], []
        for m in self.members:
            mu, log_var = m(x)
            means.append(mu)
            variances.append(torch.exp(log_var) if log_var is not None
                             else torch.zeros_like(mu))

        means = torch.stack(means, dim=0)        # (M, batch, out)
        variances = torch.stack(variances, dim=0)

        pred_mean = means.mean(dim=0)
        aleatoric_var = variances.mean(dim=0)
        epistemic_var = means.var(dim=0)
        total_std = torch.sqrt(aleatoric_var + epistemic_var + 1e-6)
        return pred_mean, total_std, None

    @torch.no_grad()
    def predict_decomposed(
        self, x: Tensor, n_samples: int = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predictive mean with epistemic and aleatoric std separated."""
        self.eval()
        means, variances = [], []
        for m in self.members:
            mu, log_var = m(x)
            means.append(mu)
            variances.append(torch.exp(log_var) if log_var is not None
                             else torch.zeros_like(mu))

        means = torch.stack(means, dim=0)
        variances = torch.stack(variances, dim=0)

        pred_mean = means.mean(dim=0)
        epistemic_std = means.std(dim=0)
        aleatoric_std = torch.sqrt(variances.mean(dim=0) + 1e-6)
        return pred_mean, epistemic_std, aleatoric_std

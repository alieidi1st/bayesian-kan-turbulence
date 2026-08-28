"""
Bayesian Kolmogorov-Arnold Network for uncertainty quantification.
"""

import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .bayesian_kan_layer import BayesianKANLayer


class BayesianKAN(nn.Module):
    """
    Bayesian Kolmogorov-Arnold Network using variational inference.

    The architecture consists of stacked BayesianKANLayer blocks, where each
    layer applies learnable B-spline activations on every edge with Gaussian
    posteriors over the spline coefficients (B-coef variant). An optional
    second output head models input-dependent (heteroscedastic) aleatoric
    uncertainty.

    Uncertainty is decomposed into two components:

    * Epistemic uncertainty: variance of predictions across posterior samples.
      Arises from coefficient uncertainty; reduced by additional data.

    * Aleatoric uncertainty: mean predicted variance from the noise head.
      Captures irreducible data noise; cannot be reduced by more data.

    Training minimises the negative Evidence Lower Bound (ELBO):

        L = -E_q[log p(y | x, theta)]  +  beta * KL(q(theta) || p(theta))

    where q(theta) is the variational posterior, p(theta) = N(0, prior_std^2)
    is the prior, and beta is an annealing coefficient.

    This class has the same calling interface as BayesianNN so that
    BNNTrainer can be used without modification.

    Parameters
    ----------
    input_dim : int
        Dimension of input features. Default: 1.
    hidden_dims : list of int
        Number of nodes in each hidden KAN layer. Default: [8, 8].
    output_dim : int
        Dimension of the prediction output. Default: 1.
    num : int
        Number of B-spline grid intervals per layer. Default: 5.
    k : int
        B-spline polynomial order. Default: 3.
    prior_std : float
        Standard deviation of the isotropic Gaussian prior on all B-spline
        coefficients across all layers. Default: 1.0.
    learn_noise : bool
        If True, a second output KAN layer predicts log-variance of the
        aleatoric noise (heteroscedastic likelihood). Default: True.
    coef_log_var_init : float
        Initial log-variance for all B-spline coefficient posteriors.
        Default: -5.0.
    grid_range : list of float
        Initial grid domain [a, b] for all KAN layers. Default: [-1, 1].
    log_var_min : float
        Lower bound applied to the noise log-variance output. Prevents
        numerical underflow from near-zero predicted variance.
        Default: -20.0 (sigma_min ≈ 2e-5).
    log_var_max : float
        Upper bound applied to the noise log-variance output. Prevents
        catastrophic blow-up when the noise head extrapolates outside the
        training distribution. Default: 5.0 (sigma_max ≈ 12).
    base_fun : {'silu', None} or nn.Module
        Residual activation function applied on each edge alongside the
        spline. Default: 'silu' (matches pykan / Kalia 2025). Set to None
        for pure-spline mode, in which each activation is a single B-spline
        with no residual branch. Pure-spline mode is recommended for symbolic
        interpretability, where the spline alone must carry the signal.
    device : str
        Torch device string. Default: 'cpu'.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        num: int = 5,
        k: int = 3,
        prior_std: float = 1.0,
        prior_type: str = 'gaussian',
        gamma_a0: float = 1e-3,
        gamma_b0: float = 1e-3,
        variant: str = 'coef',
        learn_noise: bool = True,
        noise: str = 'heteroscedastic',
        coef_log_var_init: float = -5.0,
        grid_range: list = None,
        log_var_min: float = -20.0,
        log_var_max: float = 5.0,
        base_fun='silu',
        device: str = 'cpu',
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [8, 8]
        if grid_range is None:
            grid_range = [-1, 1]
        if base_fun == 'silu':
            base_fun = nn.SiLU()

        # Noise model: 'heteroscedastic' (sigma(x) head; Kendall & Gal 2017),
        # 'homoscedastic' (single global noise variance; faithful to the
        # original BNN papers), or None (no aleatoric term). learn_noise=False
        # disables the aleatoric term entirely (used for interpretability).
        self.noise = noise if learn_noise else None
        self.learn_noise = self.noise is not None

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        self.variant = variant
        layer_kwargs = dict(
            num=num,
            k=k,
            base_fun=base_fun,
            prior_std=prior_std,
            prior_type=prior_type,
            gamma_a0=gamma_a0,
            gamma_b0=gamma_b0,
            variant=variant,
            coef_log_var_init=coef_log_var_init,
            grid_range=grid_range,
            device=device,
        )

        # ── Hidden KAN layers ─────────────────────────────────────────────────
        dims = [input_dim] + hidden_dims
        self.hidden_layers = nn.ModuleList([
            BayesianKANLayer(in_dim=d_in, out_dim=d_out, **layer_kwargs)
            for d_in, d_out in zip(dims[:-1], dims[1:])
        ])

        # ── Output layer: predicted mean ──────────────────────────────────────
        self.output_layer = BayesianKANLayer(
            in_dim=dims[-1], out_dim=output_dim, **layer_kwargs
        )

        # ── Aleatoric noise model ─────────────────────────────────────────────
        if self.noise == 'heteroscedastic':
            self.noise_layer = BayesianKANLayer(
                in_dim=dims[-1], out_dim=output_dim, **layer_kwargs
            )
        elif self.noise == 'homoscedastic':
            self.global_log_var = nn.Parameter(torch.zeros(output_dim))

        self.to(device)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self, x: Tensor, sample: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through the Bayesian KAN.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)
            Input features.
        sample : bool
            If True, draw from the posterior (stochastic). If False,
            use mean coefficients (MAP estimate). Default: True.

        Returns
        -------
        mean : Tensor, shape (batch, output_dim)
            Predicted mean.
        log_var : Tensor or None, shape (batch, output_dim)
            Predicted log-variance of the observation noise, or None
            if learn_noise is False.
        """
        h = x
        for layer in self.hidden_layers:
            h, _, _, _ = layer(h, sample=sample)

        mean, _, _, _ = self.output_layer(h, sample=sample)

        if self.noise == 'heteroscedastic':
            log_var_raw, _, _, _ = self.noise_layer(h, sample=sample)
            log_var = torch.clamp(log_var_raw, min=self.log_var_min, max=self.log_var_max)
            return mean, log_var
        elif self.noise == 'homoscedastic':
            log_var = torch.clamp(self.global_log_var, self.log_var_min,
                                  self.log_var_max).expand_as(mean)
            return mean, log_var

        return mean, None

    # ── KL divergence ─────────────────────────────────────────────────────────

    def kl_divergence(self) -> Tensor:
        """
        Total KL divergence summed over all variational layers.

        Aggregates contributions from all hidden BayesianKANLayer blocks
        and both output heads.

        Returns
        -------
        kl : Tensor
            Scalar total KL divergence.
        """
        device = next(self.parameters()).device
        kl = torch.zeros(1, device=device).squeeze()

        for layer in self.hidden_layers:
            kl = kl + layer.kl_divergence()

        kl = kl + self.output_layer.kl_divergence()

        if self.noise == 'heteroscedastic':
            kl = kl + self.noise_layer.kl_divergence()

        return kl

    # ── ELBO loss ─────────────────────────────────────────────────────────────

    def elbo_loss(
        self,
        x: Tensor,
        y: Tensor,
        n_samples: int = 1,
        kl_weight: float = 1.0,
        dataset_size: int = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Evidence Lower Bound (ELBO) loss.

        The loss is:

            L = NLL  +  kl_weight * KL / N

        where NLL is estimated with n_samples Monte Carlo draws of the
        posterior and the KL is divided by the full training-set size N
        (dataset_size). This is the correct minibatch ELBO scaling; using the
        batch size would over-weight the KL by N/batch. Falls back to the batch
        size when dataset_size is not supplied.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)
        y : Tensor, shape (batch, output_dim)
        n_samples : int
            Number of MC posterior samples for the NLL estimate. Default: 1.
        kl_weight : float
            Annealing coefficient for the KL term. Default: 1.0.

        Returns
        -------
        loss : Tensor
            Scalar negative ELBO (minimise this).
        nll : Tensor
            Scalar negative log-likelihood (MC average over n_samples).
        kl : Tensor
            Scalar KL divergence normalised by batch size.
        """
        nll = torch.zeros(1, device=x.device).squeeze()

        for _ in range(n_samples):
            mean, log_var = self.forward(x, sample=True)

            if self.learn_noise:
                # Gaussian NLL (homoscedastic or heteroscedastic):
                # -log p(y|x,theta) = 0.5 * [log(2pi) + log_var + (y-mean)^2/var]
                var = torch.exp(log_var) + 1e-6
                nll = nll + 0.5 * torch.mean(
                    math.log(2 * math.pi) + log_var + (y - mean).pow(2) / var)
            else:
                nll = nll + 0.5 * torch.mean((y - mean).pow(2))

        nll = nll / n_samples
        kl = self.kl_divergence() / (dataset_size or x.shape[0])

        loss = nll + kl_weight * kl
        return loss, nll, kl

    # ── Inference ─────────────────────────────────────────────────────────────

    def _forward_hidden(self, x: Tensor, sample: bool) -> Tensor:
        """Pass x through all hidden layers and the output layer."""
        h = x
        for layer in self.hidden_layers:
            h, _, _, _ = layer(h, sample=sample)
        mean, _, _, _ = self.output_layer(h, sample=sample)
        return mean

    def _aleatoric_std_map(self, x: Tensor) -> Tensor:
        """
        Aleatoric standard deviation.

        * homoscedastic : the single global learned noise std (broadcast).
        * heteroscedastic : sigma(x) evaluated at the posterior mean of all
          parameters (MAP). Using the MAP rather than averaging over noise-head
          samples avoids instability from B-spline extrapolation outside the
          training domain, where individual samples can produce extreme
          log-variance values whose exponentials dominate the sample mean.
        """
        if self.noise == 'homoscedastic':
            log_var = torch.clamp(self.global_log_var, self.log_var_min,
                                  self.log_var_max)
            return torch.sqrt(torch.exp(log_var) + 1e-6).expand(
                x.shape[0], self.output_dim)

        h = x
        for layer in self.hidden_layers:
            h, _, _, _ = layer(h, sample=False)
        log_var_map, _, _, _ = self.noise_layer(h, sample=False)
        log_var_map = torch.clamp(log_var_map, min=self.log_var_min, max=self.log_var_max)
        return torch.sqrt(torch.exp(log_var_map) + 1e-6)

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        n_samples: int = 100,
        return_individual: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Predictive mean and total uncertainty via Monte Carlo integration.

        Total predictive variance:

            Var_total = Var_epistemic  +  Var_aleatoric

        Epistemic variance = variance of mean predictions across n_samples
        posterior draws of the hidden and output layer coefficients.

        Aleatoric variance = exp(log_var) evaluated at the posterior mean
        of the noise head (MAP estimate). This avoids instability from
        averaging exp(log_var) over stochastic noise-head samples, which is
        unreliable for B-spline extrapolation outside the training domain.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)
        n_samples : int
            Number of posterior samples. Default: 100.
        return_individual : bool
            If True, also return the stacked per-sample mean predictions.
            Default: False.

        Returns
        -------
        mean : Tensor, shape (batch, output_dim)
            Predictive mean.
        std : Tensor, shape (batch, output_dim)
            Total predictive standard deviation.
        samples : Tensor or None, shape (n_samples, batch, output_dim)
        """
        self.eval()
        means = [self._forward_hidden(x, sample=True) for _ in range(n_samples)]
        means = torch.stack(means, dim=0)   # (n_samples, batch, output_dim)

        pred_mean = means.mean(dim=0)
        epistemic_var = means.var(dim=0)

        if self.learn_noise:
            aleatoric_var = self._aleatoric_std_map(x).pow(2)
            total_var = epistemic_var + aleatoric_var
        else:
            total_var = epistemic_var

        total_std = torch.sqrt(total_var + 1e-6)
        return pred_mean, total_std, (means if return_individual else None)

    @torch.no_grad()
    def predict_decomposed(
        self,
        x: Tensor,
        n_samples: int = 100,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predictive mean with epistemic and aleatoric uncertainty decomposed.

        See predict() for the decomposition strategy.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim)
        n_samples : int
            Number of posterior samples. Default: 100.

        Returns
        -------
        mean : Tensor, shape (batch, output_dim)
            Predictive mean.
        epistemic_std : Tensor, shape (batch, output_dim)
            Epistemic component (model uncertainty from coefficient posteriors).
        aleatoric_std : Tensor, shape (batch, output_dim)
            Aleatoric component (learned data noise). Zero if learn_noise=False.
        """
        self.eval()
        means = [self._forward_hidden(x, sample=True) for _ in range(n_samples)]
        means = torch.stack(means, dim=0)

        pred_mean = means.mean(dim=0)
        epistemic_std = means.std(dim=0)
        aleatoric_std = self._aleatoric_std_map(x) if self.learn_noise else torch.zeros_like(pred_mean)

        return pred_mean, epistemic_std, aleatoric_std

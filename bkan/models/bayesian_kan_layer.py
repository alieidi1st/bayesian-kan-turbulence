"""
Bayesian KAN layer with variational inference over B-spline coefficients.
"""

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from kan.spline import B_batch, coef2curve, curve2coef, extend_grid

from . import priors


class BayesianKANLayer(nn.Module):
    """
    KAN layer with variational inference over B-spline coefficients (B-coef variant).

    Each B-spline coefficient c_{i,j,k} on edge (input i -> output j) is
    treated as a Gaussian random variable with a learnable posterior:

        c_{i,j,k} ~ N(mu_{i,j,k}, sigma_{i,j,k}^2)

    The activation on edge (i -> j) is:

        phi_{i,j}(x_i) = w_b[i,j] * b(x_i)  +  w_s[i,j] * s_{i,j}(x_i)

    where b(x) = SiLU(x) is the fixed residual function and

        s_{i,j}(x) = sum_k  c_{i,j,k} * B_k(x)

    is the stochastic B-spline. B_k denotes B-spline basis functions
    defined on a fixed adaptive grid. The scale parameters w_b and w_s
    are deterministic in this variant.

    The forward pass uses the local reparameterization trick, which
    propagates coefficient uncertainty analytically through the basis:

        E[s_{i,j}(x_i)]   = sum_k  B_k(x_i) * mu_{i,j,k}
        Var[s_{i,j}(x_i)] = sum_k  B_k(x_i)^2 * sigma_{i,j,k}^2

    and then draws a single sample per activation:

        s_{i,j}(x_i) = E[...] + sqrt(Var[...]) * epsilon,  epsilon ~ N(0,1)

    This is equivalent to sampling coefficients and evaluating the spline,
    but produces lower-variance gradient estimates.

    The layer output for node j is the KA summation over all inputs:

        y_j = sum_i  phi_{i,j}(x_i)

    Parameters
    ----------
    in_dim : int
        Number of input features. Default: 3.
    out_dim : int
        Number of output features. Default: 2.
    num : int
        Number of B-spline grid intervals G. Default: 5.
    k : int
        B-spline polynomial order. Default: 3.
    noise_scale : float
        Amplitude of random noise used to initialise coef_mu. Default: 0.5.
    scale_base_mu : float
        Mean of the initialisation distribution for scale_base. Default: 0.0.
    scale_base_sigma : float
        Standard deviation of the initialisation distribution for scale_base.
        Default: 1.0.
    scale_sp : float
        Initial value of the spline amplitude. Default: 1.0.
    base_fun : nn.Module
        Residual activation function b(x). Default: SiLU.
    grid_eps : float
        Blending coefficient between adaptive (0) and uniform (1) grid
        during grid updates. Default: 0.02.
    grid_range : list of float
        Initial domain [a, b] of the B-spline grid. Default: [-1, 1].
    prior_std : float
        Standard deviation of the isotropic Gaussian prior on all
        B-spline coefficients. Default: 1.0.
    coef_log_var_init : float
        Constant initial value of log sigma^2 for all coefficients. A large
        negative value starts training close to the deterministic KAN limit.
        Default: -5.0.
    sp_trainable : bool
        Whether scale_sp is a learnable parameter. Default: True.
    sb_trainable : bool
        Whether scale_base is a learnable parameter. Default: True.
    device : str
        Torch device. Default: 'cpu'.
    """

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 2,
        num: int = 5,
        k: int = 3,
        noise_scale: float = 0.5,
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        base_fun: nn.Module = nn.SiLU(),
        grid_eps: float = 0.02,
        grid_range: list = None,
        prior_std: float = 1.0,
        prior_type: str = 'gaussian',
        gamma_a0: float = 1e-3,
        gamma_b0: float = 1e-3,
        variant: str = 'coef',
        coef_log_var_init: float = -5.0,
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        device: str = 'cpu',
    ):
        super().__init__()

        if grid_range is None:
            grid_range = [-1, 1]
        if variant not in ('coef', 'scale', 'full'):
            raise ValueError("variant must be 'coef', 'scale', or 'full'")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k
        self.prior_std = prior_std
        self.prior_type = prior_type
        self.gamma_a0 = gamma_a0
        self.gamma_b0 = gamma_b0
        self.base_fun = base_fun
        self.grid_eps = grid_eps

        # Which parameter groups carry a Gaussian posterior:
        #   'coef'  -> spline coefficients only (B-coef)
        #   'scale' -> edge amplitudes scale_sp / scale_base only (B-scale)
        #   'full'  -> both (B-full)
        self.variant = variant
        self.coef_variational = variant in ('coef', 'full')
        self.scale_variational = variant in ('scale', 'full')

        # ── Grid ─────────────────────────────────────────────────────────────
        # Non-trainable; updated adaptively via update_grid_from_samples.
        # Shape: (in_dim, G + 2k)
        grid = torch.linspace(
            grid_range[0], grid_range[1], steps=num + 1
        )[None, :].expand(in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = nn.Parameter(grid, requires_grad=False)

        # ── B-spline coefficient posterior ────────────────────────────────────
        # Shape of both parameters: (in_dim, out_dim, G + k)
        noises = (torch.rand(num + 1, in_dim, out_dim) - 0.5) * noise_scale / num
        coef_init = curve2coef(
            self.grid[:, k:-k].permute(1, 0), noises, self.grid, k
        )
        self.coef_mu = nn.Parameter(coef_init)
        # coef_log_var is trained only when the coefficients are variational.
        self.coef_log_var = nn.Parameter(
            torch.full_like(coef_init, coef_log_var_init),
            requires_grad=self.coef_variational,
        )

        # ── Scale (edge-amplitude) parameters ─────────────────────────────────
        # Shape: (in_dim, out_dim). These are the posterior means; the matching
        # log-variances below are trained only in the 'scale'/'full' variants.
        # When base_fun is None (pure-spline mode) the residual branch is
        # disabled: scale_base is frozen at zero so each activation is a pure
        # B-spline (used for symbolic interpretability).
        if base_fun is None:
            self.scale_base = nn.Parameter(
                torch.zeros(in_dim, out_dim), requires_grad=False
            )
        else:
            self.scale_base = nn.Parameter(
                scale_base_mu / math.sqrt(in_dim)
                + scale_base_sigma
                * (torch.rand(in_dim, out_dim) * 2.0 - 1.0)
                / math.sqrt(in_dim),
                requires_grad=sb_trainable,
            )
        self.scale_base_log_var = nn.Parameter(
            torch.full((in_dim, out_dim), coef_log_var_init),
            requires_grad=self.scale_variational and base_fun is not None,
        )
        self.scale_sp_log_var = nn.Parameter(
            torch.full((in_dim, out_dim), coef_log_var_init),
            requires_grad=self.scale_variational,
        )
        self.scale_sp = nn.Parameter(
            torch.ones(in_dim, out_dim) * scale_sp / math.sqrt(in_dim),
            requires_grad=sp_trainable,
        )

        # ── Pruning mask (non-trainable; set externally) ─────────────────────
        # Shape: (in_dim, out_dim)
        self.mask = nn.Parameter(
            torch.ones(in_dim, out_dim), requires_grad=False
        )

        self.to(device)

    # ── Device handling ───────────────────────────────────────────────────────

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self, x: Tensor, sample: bool = True
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward pass through the Bayesian KAN layer.

        Parameters
        ----------
        x : Tensor, shape (batch, in_dim)
            Layer input.
        sample : bool
            If True, draw a stochastic sample using the local
            reparameterization trick. If False, use mean coefficients
            (deterministic MAP forward pass). Default: True.

        Returns
        -------
        y : Tensor, shape (batch, out_dim)
            Layer output.
        preacts : Tensor, shape (batch, out_dim, in_dim)
            Input values broadcast to every output node (for visualisation).
        postacts : Tensor, shape (batch, out_dim, in_dim)
            Per-edge activation values after scaling and masking.
        postspline : Tensor, shape (batch, out_dim, in_dim)
            Per-edge spline-branch output using mean coefficients
            (for visualisation and attribution scoring).
        """
        batch = x.shape[0]
        preacts = x[:, None, :].clone().expand(batch, self.out_dim, self.in_dim)

        # Residual branch: b(x), shape (batch, in_dim). Disabled (pure spline)
        # when base_fun is None.
        base = self.base_fun(x) if self.base_fun is not None else None

        # B-spline basis: B[b, i, k] = B_k(x[b, i])
        # Shape: (batch, in_dim, G + k)
        B = B_batch(x, self.grid, k=self.k)

        # Mean spline output: E[s_{i,j}(x_i)] = sum_k B_k(x_i) * mu_{i,j,k}
        # Shape: (batch, in_dim, out_dim)
        y_mean = torch.einsum('bik,iok->bio', B, self.coef_mu)

        if sample and self.coef_variational:
            # Local reparameterization for coefficients:
            # Var[s_{i,j}(x_i)] = sum_k B_k(x_i)^2 * sigma_{i,j,k}^2
            coef_var = torch.exp(self.coef_log_var)
            y_var = torch.einsum('bik,iok->bio', B.pow(2), coef_var)
            y_spline = y_mean + torch.sqrt(y_var + 1e-8) * torch.randn_like(y_mean)
        else:
            y_spline = y_mean

        # Spline-only output (mean) for visualisation
        # Shape: (batch, out_dim, in_dim)
        postspline = y_mean.permute(0, 2, 1).clone()

        # Edge amplitudes: sampled in the 'scale'/'full' variants, else the mean.
        if sample and self.scale_variational:
            scale_sp = self.scale_sp + torch.exp(0.5 * self.scale_sp_log_var) * torch.randn_like(self.scale_sp)
            scale_base = self.scale_base
            if self.base_fun is not None:
                scale_base = self.scale_base + torch.exp(0.5 * self.scale_base_log_var) * torch.randn_like(self.scale_base)
        else:
            scale_sp, scale_base = self.scale_sp, self.scale_base

        # Combine residual and spline branches, apply scales and mask
        # Shape: (batch, in_dim, out_dim)
        y = scale_sp[None, :, :] * y_spline
        if base is not None:
            y = y + scale_base[None, :, :] * base[:, :, None]
        y = self.mask[None, :, :] * y

        postacts = y.permute(0, 2, 1).clone()

        # KA summation: y_j = sum_i phi_{i,j}(x_i)
        # Shape: (batch, out_dim)
        y = torch.sum(y, dim=1)

        return y, preacts, postacts, postspline

    # ── KL divergence ─────────────────────────────────────────────────────────

    def _group_kl(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """KL of one variational parameter group against its prior."""
        var = torch.exp(log_var)
        if self.prior_type == 'gamma':
            with torch.no_grad():
                sq_sum = (mu.pow(2) + var).sum()
                prior_var = priors.gamma_adaptive_prior_variance(
                    sq_sum, mu.numel(), self.gamma_a0, self.gamma_b0)
        else:
            prior_var = self.prior_std ** 2
        return priors.gaussian_kl(mu, var, prior_var)

    def kl_divergence(self) -> Tensor:
        """
        KL divergence between the posterior and the prior, summed over the
        variational parameter groups selected by `variant`:

        * 'coef'  : spline coefficients only.
        * 'scale' : edge amplitudes scale_sp (and scale_base if a residual
                    branch is present) only.
        * 'full'  : coefficients and amplitudes.

        Each group's prior is N(0, prior_var) with prior_var fixed
        (prior_type='gaussian') or set by the Gamma/ARD update
        (prior_type='gamma').
        """
        device = self.coef_mu.device
        kl = torch.zeros((), device=device)

        if self.coef_variational:
            kl = kl + self._group_kl(self.coef_mu, self.coef_log_var)

        if self.scale_variational:
            kl = kl + self._group_kl(self.scale_sp, self.scale_sp_log_var)
            if self.base_fun is not None:
                kl = kl + self._group_kl(self.scale_base, self.scale_base_log_var)

        return kl

    # ── Grid adaptation ───────────────────────────────────────────────────────

    def update_grid_from_samples(self, x: Tensor, mode: str = 'sample') -> None:
        """
        Update B-spline grid knots based on the empirical distribution of x.

        The mean coefficients coef_mu are re-fitted on the new grid so that
        the mean activation functions are preserved. The log-variance is
        re-initialised at its previous mean value to maintain the overall
        uncertainty scale after re-parameterisation.

        Parameters
        ----------
        x : Tensor, shape (batch, in_dim)
            Sample inputs used to determine the new grid.
        mode : str
            'sample': percentile-based adaptive grid placement.
            'grid': uniform spacing on a finer intermediate grid.
            Default: 'sample'.
        """
        batch = x.shape[0]
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef_mu, self.k)
        num_interval = self.grid.shape[1] - 1 - 2 * self.k

        def _make_grid(n_intervals: int) -> Tensor:
            ids = [int(batch / n_intervals * i) for i in range(n_intervals)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1, 0)
            margin = 1e-4
            h = (
                grid_adaptive[:, [-1]] - grid_adaptive[:, [0]] + 2.0 * margin
            ) / n_intervals
            grid_uniform = (
                grid_adaptive[:, [0]] - margin
                + h * torch.arange(n_intervals + 1, device=x.device)[None, :]
            )
            return self.grid_eps * grid_uniform + (1.0 - self.grid_eps) * grid_adaptive

        grid = _make_grid(num_interval)

        if mode == 'grid':
            sample_grid = _make_grid(2 * num_interval)
            x_pos = sample_grid.permute(1, 0)
            y_eval = coef2curve(x_pos, self.grid, self.coef_mu, self.k)

        mean_log_var = self.coef_log_var.data.mean().item()

        self.grid.data = extend_grid(grid, k_extend=self.k)
        new_coef_mu = curve2coef(x_pos, y_eval, self.grid, self.k)
        self.coef_mu.data = new_coef_mu
        self.coef_log_var.data = torch.full_like(new_coef_mu, mean_log_var)

    # ── Interpretability ──────────────────────────────────────────────────────

    @torch.no_grad()
    def get_activation_stats(
        self,
        in_idx: int,
        out_idx: int,
        x_grid: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Mean and pointwise standard deviation of activation phi_{in,out}(x).

        Uses the local reparameterization result analytically:

            E[phi(x)]   = w_b * b(x)  +  w_s * sum_k  B_k(x) * mu_k
            Std[phi(x)] = |w_s| * sqrt( sum_k  B_k(x)^2 * sigma_k^2 )

        where b(x) = SiLU(x) is the deterministic residual function and
        w_b, w_s are the deterministic scale parameters.  Only the spline
        coefficients carry uncertainty in the B-coef variant.

        Parameters
        ----------
        in_idx : int
            Source node index (0 <= in_idx < in_dim).
        out_idx : int
            Destination node index (0 <= out_idx < out_dim).
        x_grid : Tensor, shape (n_grid,)
            1D grid of input values at which to evaluate the activation.

        Returns
        -------
        mean_phi : Tensor, shape (n_grid,)
            Mean activation function E[phi_{in,out}(x)].
        std_phi : Tensor, shape (n_grid,)
            Pointwise standard deviation Std[phi_{in,out}(x)] from the
            B-spline coefficient posterior.
        """
        x_eval = x_grid.unsqueeze(1).to(self.device)       # (n_grid, 1)
        grid_i = self.grid[[in_idx], :]                     # (1, G + 2k)

        # B-spline basis evaluated at x_grid for input dimension in_idx
        # Shape: (n_grid, 1, G+k)  →  squeeze to (n_grid, G+k)
        B = B_batch(x_eval, grid_i, k=self.k)[:, 0, :]

        mu      = self.coef_mu[in_idx, out_idx, :]          # (G+k,)
        sig_sq  = torch.exp(self.coef_log_var[in_idx, out_idx, :])  # (G+k,)

        spline_mean = B @ mu                                 # (n_grid,)
        spline_std  = torch.sqrt((B ** 2) @ sig_sq + 1e-8)  # (n_grid,)

        w_s = self.scale_sp[in_idx, out_idx]
        mean_phi = w_s * spline_mean
        std_phi  = torch.abs(w_s) * spline_std

        # Add residual branch unless in pure-spline mode (base_fun is None)
        if self.base_fun is not None:
            w_b = self.scale_base[in_idx, out_idx]
            base_val = self.base_fun(x_grid.to(self.device))   # (n_grid,)
            mean_phi = mean_phi + w_b * base_val

        return mean_phi.cpu(), std_phi.cpu()

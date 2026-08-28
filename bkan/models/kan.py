"""
Deterministic Kolmogorov-Arnold Network for regression.

A point-estimate KAN built on the same B-spline machinery as BayesianKAN, so
that the BKAN-vs-KAN comparison isolates exactly the Bayesian treatment (the
architecture, spline basis, and residual branch are identical; only the
variational posterior over coefficients is removed).

Implements the standard KAN architecture of Liu et al. (2024): each edge applies
a learnable B-spline plus a SiLU residual branch, and node activations are the
Kolmogorov-Arnold sum over incoming edges.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from kan.spline import B_batch, curve2coef, extend_grid


class DeterministicKANLayer(nn.Module):
    """
    KAN layer with point-estimate B-spline coefficients (no posterior).

    phi_{i,j}(x_i) = scale_base[i,j] * b(x_i) + scale_sp[i,j] * spline_{i,j}(x_i)
    y_j = sum_i phi_{i,j}(x_i)

    Parameters mirror BayesianKANLayer (minus the variational pieces) so the two
    are architecturally matched.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num: int = 5,
        k: int = 3,
        noise_scale: float = 0.5,
        scale_base_sigma: float = 1.0,
        scale_sp: float = 1.0,
        base_fun: Optional[nn.Module] = nn.SiLU(),
        grid_range: list = None,
        device: str = 'cpu',
    ):
        super().__init__()
        if grid_range is None:
            grid_range = [-1, 1]
        self.in_dim, self.out_dim, self.num, self.k = in_dim, out_dim, num, k
        self.base_fun = base_fun

        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(
            in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.grid = nn.Parameter(grid, requires_grad=False)

        noises = (torch.rand(num + 1, in_dim, out_dim) - 0.5) * noise_scale / num
        self.coef = nn.Parameter(curve2coef(self.grid[:, k:-k].permute(1, 0), noises,
                                            self.grid, k))

        if base_fun is None:
            self.scale_base = nn.Parameter(torch.zeros(in_dim, out_dim),
                                           requires_grad=False)
        else:
            self.scale_base = nn.Parameter(
                scale_base_sigma * (torch.rand(in_dim, out_dim) * 2 - 1) / math.sqrt(in_dim))
        self.scale_sp = nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp / math.sqrt(in_dim))
        self.to(device)

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def forward(self, x: Tensor) -> Tensor:
        B = B_batch(x, self.grid, k=self.k)                 # (batch, in_dim, G+k)
        spline = torch.einsum('bik,iok->bio', B, self.coef)  # (batch, in_dim, out_dim)
        y = self.scale_sp[None, :, :] * spline
        if self.base_fun is not None:
            y = y + self.scale_base[None, :, :] * self.base_fun(x)[:, :, None]
        return torch.sum(y, dim=1)                            # (batch, out_dim)


class DeterministicKAN(nn.Module):
    """
    Deterministic KAN regressor (point-estimate baseline for BKAN).

    Has no epistemic uncertainty (weights are point estimates). An optional
    noise model provides aleatoric uncertainty for likelihood-based metrics:
    'homoscedastic' (single global variance), 'heteroscedastic' (sigma(x) head),
    or None (mean only, MSE loss).

    Interface (loss, predict, predict_decomposed) matches the other models so the
    shared trainer and evaluation code apply. predict_decomposed returns zero
    epistemic std, by construction.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        num: int = 5,
        k: int = 3,
        noise: Optional[str] = 'homoscedastic',
        grid_range: list = None,
        base_fun='silu',
        log_var_min: float = -20.0,
        log_var_max: float = 5.0,
        device: str = 'cpu',
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [8, 8]
        if grid_range is None:
            grid_range = [-1, 1]
        if base_fun == 'silu':
            base_fun = nn.SiLU()

        self.input_dim, self.output_dim = input_dim, output_dim
        self.noise = noise
        self.log_var_min, self.log_var_max = log_var_min, log_var_max
        lk = dict(num=num, k=k, base_fun=base_fun, grid_range=grid_range, device=device)

        dims = [input_dim] + hidden_dims
        self.hidden_layers = nn.ModuleList(
            [DeterministicKANLayer(a, b, **lk) for a, b in zip(dims[:-1], dims[1:])])
        self.output_layer = DeterministicKANLayer(dims[-1], output_dim, **lk)

        if noise == 'heteroscedastic':
            self.noise_layer = DeterministicKANLayer(dims[-1], output_dim, **lk)
        elif noise == 'homoscedastic':
            self.global_log_var = nn.Parameter(torch.zeros(output_dim))
        self.to(device)

    def _hidden(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.hidden_layers:
            h = layer(h)
        return h

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        h = self._hidden(x)
        mean = self.output_layer(h)
        if self.noise == 'heteroscedastic':
            log_var = torch.clamp(self.noise_layer(h), self.log_var_min, self.log_var_max)
        elif self.noise == 'homoscedastic':
            log_var = torch.clamp(self.global_log_var, self.log_var_min,
                                  self.log_var_max).expand_as(mean)
        else:
            log_var = None
        return mean, log_var

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        mean, log_var = self.forward(x)
        if log_var is not None:
            var = torch.exp(log_var) + 1e-6
            return 0.5 * torch.mean(math.log(2 * math.pi) + log_var + (y - mean).pow(2) / var)
        return 0.5 * torch.mean((y - mean).pow(2))

    @torch.no_grad()
    def _aleatoric_std(self, x: Tensor) -> Tensor:
        _, log_var = self.forward(x)
        if log_var is None:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        return torch.sqrt(torch.exp(log_var) + 1e-6)

    @torch.no_grad()
    def predict(self, x: Tensor, n_samples: int = None
                ) -> Tuple[Tensor, Tensor, None]:
        """Mean and total std. Total std is aleatoric only (no epistemic)."""
        self.eval()
        mean, _ = self.forward(x)
        return mean, self._aleatoric_std(x), None

    @torch.no_grad()
    def predict_decomposed(self, x: Tensor, n_samples: int = None
                           ) -> Tuple[Tensor, Tensor, Tensor]:
        self.eval()
        mean, _ = self.forward(x)
        epistemic_std = torch.zeros_like(mean)   # deterministic: no epistemic uncertainty
        return mean, epistemic_std, self._aleatoric_std(x)

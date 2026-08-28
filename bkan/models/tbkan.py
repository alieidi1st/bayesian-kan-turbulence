"""
Tensor-Basis Bayesian Kolmogorov-Arnold Network (TBKAN) for turbulence
anisotropy mapping with uncertainty quantification.

A Bayesian KAN maps the scalar invariants of the normalised strain/rotation-rate
tensors to the tensor-basis coefficients g_n, and a fixed merge layer reconstructs
the Reynolds-stress anisotropy

    b_ij = sum_{n=1}^{m} g_n(lambda) T^(n)_ij ,

so the prediction is Galilean- and rotation-invariant, symmetric and traceless by
construction (Pope 1975; Ling, Kurzawski & Templeton 2016). The Bayesian treatment
gives a posterior over g_n -> a predictive distribution over b_ij, decomposed into
epistemic (posterior spread of the g_n) and aleatoric (learned observation noise)
uncertainty. A realizability penalty (McConkey et al. 2025) discourages predictions
that leave the physically attainable region of anisotropy states.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .bayesian_kan_layer import BayesianKANLayer

# Index pairs of the six unique components of a symmetric 3x3 tensor.
_SYM_IDX = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def _to6(b: Tensor) -> Tensor:
    """(batch,3,3) symmetric -> (batch,6) unique components (11,12,13,22,23,33)."""
    return torch.stack([b[:, i, j] for (i, j) in _SYM_IDX], dim=1)


def realizability_penalty(b: Tensor) -> Tensor:
    """
    Differentiable realizability penalty for the anisotropy tensor.

    The anisotropy b is realizable iff its eigenvalues lie in [-1/3, 2/3]
    (equivalently, the barycentric coordinates are non-negative; McConkey et al.
    2025). This penalises the squared violation of the eigenvalue bounds:

        R(b) = mean_points [ max(0, lambda_max - 2/3)^2 + max(0, -1/3 - lambda_min)^2 ].

    Parameters
    ----------
    b : Tensor, shape (batch, 3, 3)

    Returns
    -------
    penalty : scalar Tensor
    """
    # symmetrise for numerical safety, then eigenvalues (ascending)
    bs = 0.5 * (b + b.transpose(1, 2))
    w = torch.linalg.eigvalsh(bs)                 # (batch, 3)
    lam_min, lam_max = w[:, 0], w[:, 2]
    viol = torch.clamp(lam_max - 2.0 / 3.0, min=0.0) ** 2 \
        + torch.clamp(-1.0 / 3.0 - lam_min, min=0.0) ** 2
    return viol.mean()


class TensorBasisBKAN(nn.Module):
    """
    Bayesian KAN with a Pope tensor-basis merge layer.

    Parameters
    ----------
    n_inv : int
        Number of scalar invariant inputs (2 for the 3-term basis, 5 for 10-term).
    hidden_dims : list of int
        Hidden KAN layer widths. Default: [10].
    n_basis : int
        Number of tensor-basis coefficients g_n predicted (3 or 10). Must match
        the basis fed at forward time. Default: 3.
    noise : {'homoscedastic', 'heteroscedastic', None}
        Aleatoric noise model on the six anisotropy components. 'homoscedastic'
        learns one global log-variance per component (faithful to the original
        BNN papers); 'heteroscedastic' learns sigma(lambda) via a second KAN
        head (Kendall & Gal 2017). Default: 'homoscedastic'.
    realiz_weight : float
        Weight alpha on the realizability penalty in the data term. Default: 10.0
        (McConkey et al. 2025 use up to 1e2; tune per problem).
    num, k, prior_std, prior_type, gamma_a0, gamma_b0, variant, base_fun,
    coef_log_var_init, grid_range : passed through to BayesianKANLayer.
    """

    def __init__(
        self,
        n_inv: int = 2,
        hidden_dims: List[int] = None,
        n_basis: int = 3,
        noise: str = 'homoscedastic',
        realiz_weight: float = 10.0,
        num: int = 5,
        k: int = 3,
        prior_std: float = 1.0,
        prior_type: str = 'gaussian',
        gamma_a0: float = 1e-3,
        gamma_b0: float = 1e-3,
        variant: str = 'coef',
        base_fun='silu',
        coef_log_var_init: float = -5.0,
        grid_range: list = None,
        log_var_min: float = -20.0,
        log_var_max: float = 5.0,
        device: str = 'cpu',
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [10]
        if grid_range is None:
            # standardised invariants are roughly in [-4, 4]
            grid_range = [-4, 4]
        if base_fun == 'silu':
            base_fun = nn.SiLU()

        self.n_inv = n_inv
        self.n_basis = n_basis
        self.noise = noise
        self.realiz_weight = realiz_weight
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max
        self.variant = variant

        layer_kwargs = dict(
            num=num, k=k, base_fun=base_fun, prior_std=prior_std,
            prior_type=prior_type, gamma_a0=gamma_a0, gamma_b0=gamma_b0,
            variant=variant, coef_log_var_init=coef_log_var_init,
            grid_range=grid_range, device=device,
        )

        # ── KAN backbone: invariants -> g coefficients ────────────────────────
        dims = [n_inv] + hidden_dims
        self.hidden_layers = nn.ModuleList([
            BayesianKANLayer(in_dim=a, out_dim=b, **layer_kwargs)
            for a, b in zip(dims[:-1], dims[1:])
        ])
        self.coef_layer = BayesianKANLayer(in_dim=dims[-1], out_dim=n_basis,
                                           **layer_kwargs)

        # ── aleatoric noise on the 6 anisotropy components ────────────────────
        if noise == 'homoscedastic':
            self.global_log_var = nn.Parameter(torch.zeros(6))
        elif noise == 'heteroscedastic':
            self.noise_hidden = nn.ModuleList([
                BayesianKANLayer(in_dim=a, out_dim=b, **layer_kwargs)
                for a, b in zip(dims[:-1], dims[1:])
            ])
            self.noise_layer = BayesianKANLayer(in_dim=dims[-1], out_dim=6,
                                                **layer_kwargs)
        elif noise is not None:
            raise ValueError(f'unknown noise model {noise!r}')

        self.to(device)

    # ── forward ───────────────────────────────────────────────────────────────
    def coefficients(self, x: Tensor, sample: bool = True) -> Tensor:
        """Predict the tensor-basis coefficients g (batch, n_basis)."""
        h = x
        for layer in self.hidden_layers:
            h, _, _, _ = layer(h, sample=sample)
        g, _, _, _ = self.coef_layer(h, sample=sample)
        return g

    def forward(self, x: Tensor, T: Tensor, sample: bool = True) -> Tensor:
        """
        Reconstruct the anisotropy b = sum_n g_n T^(n).

        Parameters
        ----------
        x : Tensor (batch, n_inv)   standardised invariants
        T : Tensor (batch, n_basis, 3, 3)   precomputed basis tensors
        Returns
        -------
        b : Tensor (batch, 3, 3)
        """
        g = self.coefficients(x, sample=sample)
        return torch.einsum('bn,bnij->bij', g, T)

    def _noise_log_var(self, x: Tensor, sample: bool) -> Tensor:
        if self.noise == 'homoscedastic':
            lv = torch.clamp(self.global_log_var, self.log_var_min, self.log_var_max)
            return lv.expand(x.shape[0], 6)
        h = x
        for layer in self.noise_hidden:
            h, _, _, _ = layer(h, sample=sample)
        lv, _, _, _ = self.noise_layer(h, sample=sample)
        return torch.clamp(lv, self.log_var_min, self.log_var_max)

    # ── KL ─────────────────────────────────────────────────────────────────────
    def kl_divergence(self) -> Tensor:
        kl = self.coef_layer.kl_divergence()
        for layer in self.hidden_layers:
            kl = kl + layer.kl_divergence()
        if self.noise == 'heteroscedastic':
            kl = kl + self.noise_layer.kl_divergence()
            for layer in self.noise_hidden:
                kl = kl + layer.kl_divergence()
        return kl

    # ── ELBO ───────────────────────────────────────────────────────────────────
    def elbo_loss(
        self,
        x: Tensor,
        T: Tensor,
        b: Tensor,
        n_samples: int = 1,
        kl_weight: float = 1.0,
        dataset_size: int = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Negative ELBO for the anisotropy mapping.

            L = NLL(b | x, T)  +  alpha * R(b_pred)  +  kl_weight * KL / N

        NLL is a Gaussian likelihood over the six unique anisotropy components
        (per-component variance from the noise model). R is the realizability
        penalty. Returns (loss, nll, realiz, kl).
        """
        b6 = _to6(b)
        nll = torch.zeros((), device=x.device)
        realiz = torch.zeros((), device=x.device)

        for _ in range(n_samples):
            g = self.coefficients(x, sample=True)
            b_pred = torch.einsum('bn,bnij->bij', g, T)
            r6 = _to6(b_pred)

            if self.noise is not None:
                log_var = self._noise_log_var(x, sample=True)
                var = torch.exp(log_var) + 1e-8
                nll = nll + 0.5 * torch.mean(
                    math.log(2 * math.pi) + log_var + (b6 - r6).pow(2) / var)
            else:
                nll = nll + 0.5 * torch.mean((b6 - r6).pow(2))

            realiz = realiz + realizability_penalty(b_pred)

        nll = nll / n_samples
        realiz = realiz / n_samples
        kl = self.kl_divergence() / (dataset_size or x.shape[0])

        loss = nll + self.realiz_weight * realiz + kl_weight * kl
        return loss, nll, realiz, kl

    # ── inference ──────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self, x: Tensor, T: Tensor, n_samples: int = 100, chunk: int = 20000,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predictive anisotropy with epistemic + aleatoric uncertainty.

        The point prediction is taken at the posterior mean of the coefficients
        (MAP) -- stable and decoupled from sampling noise; the epistemic standard
        deviation is the spread of the reconstructed anisotropy across posterior
        draws, accumulated online (Welford) so no (n_samples, N, 6) tensor is
        ever materialised; and the computation is chunked over points so that the
        per-forward spline allocation stays bounded on very large test sets.

        Returns (per the six unique components)
        -------
        b_mean : (N, 6)   predictive mean anisotropy
        epi_std : (N, 6)  epistemic std (spread of b across posterior draws)
        alea_std : (N, 6) aleatoric std (learned observation noise)
        """
        self.eval()
        N = x.shape[0]
        b_mean = torch.empty(N, 6)
        epi_std = torch.empty(N, 6)
        alea_std = torch.zeros(N, 6)
        for s0 in range(0, N, chunk):
            sl = slice(s0, min(s0 + chunk, N))
            xc, Tc = x[sl], T[sl]
            torch.manual_seed(0)                   # reproducible MC sampling
            g_map = self.coefficients(xc, sample=False)
            b_mean[sl] = _to6(torch.einsum('bn,bnij->bij', g_map, Tc))
            # online mean / M2 of the sampled predictions (epistemic variance)
            mean = torch.zeros(xc.shape[0], 6)
            M2 = torch.zeros(xc.shape[0], 6)
            for s in range(n_samples):
                g = self.coefficients(xc, sample=True)
                b6 = _to6(torch.einsum('bn,bnij->bij', g, Tc))
                delta = b6 - mean
                mean = mean + delta / (s + 1)
                M2 = M2 + delta * (b6 - mean)
            denom = max(1, n_samples - 1)
            epi_std[sl] = torch.sqrt(M2 / denom + 1e-12)
            if self.noise is not None:
                log_var = self._noise_log_var(xc, sample=False)   # MAP noise
                alea_std[sl] = torch.sqrt(torch.exp(log_var) + 1e-8)
        return b_mean, epi_std, alea_std

    @torch.no_grad()
    def predict_coefficients(
        self, x: Tensor, n_samples: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        """Posterior mean and std of the basis coefficients g (for interpretability)."""
        self.eval()
        G = torch.stack([self.coefficients(x, sample=True) for _ in range(n_samples)], 0)
        return G.mean(0), G.std(0)

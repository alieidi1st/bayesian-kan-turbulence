"""
Variational Bayesian neural network baseline (mean-field VI).

A clean, self-contained implementation following the original BNN literature,
independent of any application-specific code:

* Mean-field Gaussian variational posterior with the softplus standard-deviation
  parameterisation sigma = log(1 + exp(rho))            -- Blundell et al. (2015)
* Reparameterisation-trick sampling                     -- Graves (2011)
* Three prior families, selectable via `prior_type`:
    - 'gaussian'       : single Gaussian, closed-form KL  (Graves 2011)
    - 'scale_mixture'  : two-Gaussian mixture, MC KL       (Blundell 2015)
    - 'gamma'          : Gamma hyperprior on precision/ARD (MacKay 1992)
* Two observation-noise models, selectable via `noise`:
    - 'homoscedastic'  : single global learned noise variance
                         (faithful to Neal/MacKay/Hernandez-Lobato)
    - 'heteroscedastic': input-dependent sigma(x) head
                         (extension; Kendall & Gal 2017)

The public interface (forward, kl_divergence, elbo_loss, predict,
predict_decomposed) matches BayesianKAN so the same trainer and evaluation code
serve both, enabling a controlled BKAN-vs-BNN comparison in which only the
backbone (MLP vs KAN) differs.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from . import priors


class BayesianLinear(nn.Module):
    """
    Linear layer with a mean-field Gaussian weight posterior (Bayes by Backprop).

    Weights and biases have posteriors N(mu, sigma^2) with sigma = softplus(rho).
    The KL term depends on `prior_type`:

    * 'gaussian'      : closed-form KL to N(0, prior_sigma^2).
    * 'scale_mixture' : Monte-Carlo KL = log q(w) - log p(w) using the weights
                        sampled during the forward pass (Blundell 2015).
    * 'gamma'         : closed-form KL to N(0, 1/lambda*), where lambda* is the
                        ARD precision update (MacKay 1992).

    Parameters
    ----------
    in_features, out_features : int
    prior_type : {'gaussian', 'scale_mixture', 'gamma'}
    prior_sigma : float
        Prior standard deviation for the Gaussian prior. Default: 1.0.
    sm_pi, sm_sigma1, sm_sigma2 : float
        Scale-mixture parameters (Blundell 2015 defaults: 0.5, 1.0, exp(-6)).
    gamma_a0, gamma_b0 : float
        Gamma hyperprior shape/rate (weakly informative default 1e-3).
    rho_init : float
        Initial value of rho (sigma = softplus(rho) ~= 0.049 at rho = -3).
    bias : bool
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_type: str = 'gaussian',
        prior_sigma: float = 1.0,
        sm_pi: float = 0.5,
        sm_sigma1: float = 1.0,
        sm_sigma2: float = math.exp(-6),
        gamma_a0: float = 1e-3,
        gamma_b0: float = 1e-3,
        rho_init: float = -3.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_type = prior_type
        self.prior_sigma = prior_sigma
        self.sm_pi, self.sm_sigma1, self.sm_sigma2 = sm_pi, sm_sigma1, sm_sigma2
        self.gamma_a0, self.gamma_b0 = gamma_a0, gamma_b0
        self.use_bias = bias

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho_init))
        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_rho = nn.Parameter(torch.full((out_features,), rho_init))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')

        # Cache for the Monte-Carlo KL of the scale-mixture prior
        self._mc_kl = None

    @staticmethod
    def _sigma(rho: Tensor) -> Tensor:
        return F.softplus(rho)

    def forward(self, x: Tensor, sample: bool = True) -> Tensor:
        if sample:
            w_sigma = self._sigma(self.weight_rho)
            weight = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
            if self.use_bias:
                b_sigma = self._sigma(self.bias_rho)
                bias = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
            else:
                bias = None

            if self.prior_type == 'scale_mixture':
                self._mc_kl = self._scale_mixture_kl(weight, w_sigma, bias,
                                                     b_sigma if self.use_bias else None)
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(x, weight, bias)

    def _scale_mixture_kl(self, weight, w_sigma, bias, b_sigma) -> Tensor:
        """MC estimate log q(w) - log p_mix(w) on the sampled weights."""
        log_q = priors.gaussian_posterior_log_prob(weight, self.weight_mu, w_sigma)
        log_p = priors.scale_mixture_log_prob(weight, self.sm_pi,
                                              self.sm_sigma1, self.sm_sigma2)
        if self.use_bias:
            log_q = log_q + priors.gaussian_posterior_log_prob(bias, self.bias_mu, b_sigma)
            log_p = log_p + priors.scale_mixture_log_prob(bias, self.sm_pi,
                                                          self.sm_sigma1, self.sm_sigma2)
        return log_q - log_p

    def kl_divergence(self) -> Tensor:
        if self.prior_type == 'scale_mixture':
            if self._mc_kl is None:
                raise RuntimeError("scale_mixture KL requires a preceding forward(sample=True)")
            return self._mc_kl

        w_var = self._sigma(self.weight_rho).pow(2)
        mus, vars_ = [self.weight_mu], [w_var]
        if self.use_bias:
            mus.append(self.bias_mu)
            vars_.append(self._sigma(self.bias_rho).pow(2))

        if self.prior_type == 'gamma':
            with torch.no_grad():
                sq_sum = sum((m.pow(2) + v).sum() for m, v in zip(mus, vars_))
                n = sum(m.numel() for m in mus)
                prior_var = priors.gamma_adaptive_prior_variance(
                    sq_sum, n, self.gamma_a0, self.gamma_b0)
        else:  # gaussian
            prior_var = self.prior_sigma ** 2

        return sum(priors.gaussian_kl(m, v, prior_var) for m, v in zip(mus, vars_))


class BNN(nn.Module):
    """
    Mean-field variational Bayesian neural network for regression.

    Parameters
    ----------
    input_dim, output_dim : int
    hidden_dims : list of int
        Hidden layer widths. Default: [50] (Hernandez-Lobato & Adams 2015).
    activation : {'relu', 'tanh', 'elu'}
        Default: 'relu' (the Hernandez-Lobato setting).
    prior_type : {'gaussian', 'scale_mixture', 'gamma'}
        Weight prior family. Default: 'gaussian'.
    noise : {'homoscedastic', 'heteroscedastic'}
        Observation-noise model. 'homoscedastic' learns a single global noise
        variance (faithful to the original BNN papers / the Hernandez-Lobato
        benchmark); 'heteroscedastic' adds a sigma(x) head (Kendall & Gal 2017).
        Default: 'homoscedastic'.
    prior_sigma, sm_*, gamma_* :
        Prior hyperparameters passed to each BayesianLinear.
    log_var_min, log_var_max : float
        Clamp range for predicted/learned log-variance.
    device : str
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        activation: str = 'relu',
        prior_type: str = 'gaussian',
        noise: str = 'homoscedastic',
        prior_sigma: float = 1.0,
        sm_pi: float = 0.5,
        sm_sigma1: float = 1.0,
        sm_sigma2: float = math.exp(-6),
        gamma_a0: float = 1e-3,
        gamma_b0: float = 1e-3,
        log_var_min: float = -20.0,
        log_var_max: float = 5.0,
        device: str = 'cpu',
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [50]
        assert noise in ('homoscedastic', 'heteroscedastic')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.noise = noise
        self.learn_noise = True  # always model aleatoric (global or input-dependent)
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        self.act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'elu': nn.ELU()}.get(
            activation, nn.ReLU())

        lk = dict(prior_type=prior_type, prior_sigma=prior_sigma, sm_pi=sm_pi,
                  sm_sigma1=sm_sigma1, sm_sigma2=sm_sigma2,
                  gamma_a0=gamma_a0, gamma_b0=gamma_b0)

        dims = [input_dim] + hidden_dims
        self.hidden_layers = nn.ModuleList(
            [BayesianLinear(a, b, **lk) for a, b in zip(dims[:-1], dims[1:])])
        self.mean_head = BayesianLinear(dims[-1], output_dim, **lk)

        if noise == 'heteroscedastic':
            self.noise_head = BayesianLinear(dims[-1], output_dim, **lk)
        else:
            # single global log noise variance (per output dimension)
            self.global_log_var = nn.Parameter(torch.zeros(output_dim))

        self.to(device)

    def forward(self, x: Tensor, sample: bool = True
                ) -> Tuple[Tensor, Tensor]:
        h = x
        for layer in self.hidden_layers:
            h = self.act(layer(h, sample=sample))
        mean = self.mean_head(h, sample=sample)

        if self.noise == 'heteroscedastic':
            log_var = torch.clamp(self.noise_head(h, sample=sample),
                                  self.log_var_min, self.log_var_max)
        else:
            log_var = torch.clamp(self.global_log_var, self.log_var_min,
                                  self.log_var_max).expand_as(mean)
        return mean, log_var

    def kl_divergence(self) -> Tensor:
        kl = sum(layer.kl_divergence() for layer in self.hidden_layers)
        kl = kl + self.mean_head.kl_divergence()
        if self.noise == 'heteroscedastic':
            kl = kl + self.noise_head.kl_divergence()
        return kl

    def elbo_loss(self, x: Tensor, y: Tensor, n_samples: int = 1,
                  kl_weight: float = 1.0, dataset_size: int = None
                  ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Negative ELBO on a (mini)batch.

        The KL term is divided by the full training-set size `dataset_size`
        (not the batch size), which is the correct minibatch ELBO scaling:
        per-batch loss = mean_batch NLL + KL / N. Passing the batch size here
        would over-weight the KL by N/batch and over-regularise the model.
        Falls back to the batch size if dataset_size is not supplied.
        """
        nll = torch.zeros(1, device=x.device).squeeze()
        for _ in range(n_samples):
            mean, log_var = self.forward(x, sample=True)
            var = torch.exp(log_var) + 1e-6
            nll = nll + 0.5 * torch.mean(
                math.log(2 * math.pi) + log_var + (y - mean).pow(2) / var)
        nll = nll / n_samples
        kl = self.kl_divergence() / (dataset_size or x.shape[0])
        return nll + kl_weight * kl, nll, kl

    @torch.no_grad()
    def _global_aleatoric_std(self, x: Tensor) -> Tensor:
        log_var = torch.clamp(self.global_log_var, self.log_var_min, self.log_var_max)
        return torch.sqrt(torch.exp(log_var) + 1e-6).expand(x.shape[0], self.output_dim)

    @torch.no_grad()
    def _heteroscedastic_aleatoric_std(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.hidden_layers:
            h = self.act(layer(h, sample=False))
        log_var = torch.clamp(self.noise_head(h, sample=False),
                              self.log_var_min, self.log_var_max)
        return torch.sqrt(torch.exp(log_var) + 1e-6)

    @torch.no_grad()
    def _mean_only(self, x: Tensor, sample: bool) -> Tensor:
        h = x
        for layer in self.hidden_layers:
            h = self.act(layer(h, sample=sample))
        return self.mean_head(h, sample=sample)

    @torch.no_grad()
    def predict(self, x: Tensor, n_samples: int = 100,
                return_individual: bool = False
                ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        self.eval()
        means = torch.stack([self._mean_only(x, sample=True) for _ in range(n_samples)], 0)
        pred_mean = means.mean(0)
        epistemic_var = means.var(0)
        aleatoric_std = (self._heteroscedastic_aleatoric_std(x)
                         if self.noise == 'heteroscedastic'
                         else self._global_aleatoric_std(x))
        total_std = torch.sqrt(epistemic_var + aleatoric_std.pow(2) + 1e-6)
        return pred_mean, total_std, (means if return_individual else None)

    @torch.no_grad()
    def predict_decomposed(self, x: Tensor, n_samples: int = 100
                           ) -> Tuple[Tensor, Tensor, Tensor]:
        self.eval()
        means = torch.stack([self._mean_only(x, sample=True) for _ in range(n_samples)], 0)
        pred_mean = means.mean(0)
        epistemic_std = means.std(0)
        aleatoric_std = (self._heteroscedastic_aleatoric_std(x)
                         if self.noise == 'heteroscedastic'
                         else self._global_aleatoric_std(x))
        return pred_mean, epistemic_std, aleatoric_std

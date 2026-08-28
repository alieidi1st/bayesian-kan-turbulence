"""
Bayesian Neural Network implementations for uncertainty quantification.

Provides two main approaches:
1. Variational Inference BNN (weight distributions)
2. MC Dropout BNN (dropout at test time)

Both support heteroscedastic noise modeling (learned aleatoric uncertainty).
"""

import math
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VariationalLinear(nn.Module):
    """
    Linear layer with variational weight distributions.

    Implements the local reparameterization trick for efficient
    gradient estimation. Weights are sampled from Gaussian posteriors
    with learnable mean and variance.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        bias: bool = True
    ):
        """
        Args:
            in_features: Size of input features
            out_features: Size of output features
            prior_std: Standard deviation of the prior distribution
            bias: Whether to include bias terms
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.use_bias = bias

        # Weight posterior parameters (mean and log-variance)
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_log_var = nn.Parameter(torch.zeros(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_log_var = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_var', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters using Kaiming initialization for means."""
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.weight_log_var, -5.0)  # Start with small variance

        if self.use_bias:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_log_var, -5.0)

    def forward(self, x: Tensor, sample: bool = True) -> Tensor:
        """
        Forward pass with optional weight sampling.

        Args:
            x: Input tensor of shape (batch_size, in_features)
            sample: If True, sample weights; if False, use mean weights

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        if sample:
            # Local reparameterization trick
            weight_std = torch.exp(0.5 * self.weight_log_var)
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

            if self.use_bias:
                bias_std = torch.exp(0.5 * self.bias_log_var)
                bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
            else:
                bias = None
        else:
            weight = self.weight_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> Tensor:
        """
        Compute KL divergence between posterior and prior.

        KL(q(w) || p(w)) for Gaussian distributions.

        Returns:
            Scalar KL divergence
        """
        prior_var = self.prior_std ** 2

        # KL for weights
        weight_var = torch.exp(self.weight_log_var)
        kl = 0.5 * torch.sum(
            weight_var / prior_var +
            self.weight_mu ** 2 / prior_var -
            1 -
            self.weight_log_var +
            math.log(prior_var)
        )

        # KL for biases
        if self.use_bias:
            bias_var = torch.exp(self.bias_log_var)
            kl += 0.5 * torch.sum(
                bias_var / prior_var +
                self.bias_mu ** 2 / prior_var -
                1 -
                self.bias_log_var +
                math.log(prior_var)
            )

        return kl


class BayesianNN(nn.Module):
    """
    Bayesian Neural Network using Variational Inference.

    Learns both epistemic uncertainty (from weight distributions) and
    optionally aleatoric uncertainty (heteroscedastic noise).
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = [50, 50],
        output_dim: int = 1,
        prior_std: float = 1.0,
        learn_noise: bool = True,
        activation: str = "tanh"
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output
            prior_std: Prior standard deviation for weights
            learn_noise: If True, learn heteroscedastic noise (aleatoric)
            activation: Activation function ("relu", "tanh", "elu")
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learn_noise = learn_noise

        # Build network layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(VariationalLinear(prev_dim, hidden_dim, prior_std))
            prev_dim = hidden_dim
        self.hidden_layers = nn.ModuleList(layers)

        # Output layer for mean prediction
        self.output_layer = VariationalLinear(prev_dim, output_dim, prior_std)

        # Optional: separate head for log-variance (heteroscedastic noise)
        if learn_noise:
            self.noise_layer = VariationalLinear(prev_dim, output_dim, prior_std)

        # Activation function
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        self.activation = activations.get(activation, nn.Tanh())

    def forward(
        self,
        x: Tensor,
        sample: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            sample: If True, sample from weight posteriors

        Returns:
            mean: Predicted mean of shape (batch_size, output_dim)
            log_var: Predicted log-variance (if learn_noise=True), else None
        """
        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h, sample=sample))

        mean = self.output_layer(h, sample=sample)

        if self.learn_noise:
            log_var = self.noise_layer(h, sample=sample)
            return mean, log_var
        else:
            return mean, None

    def kl_divergence(self) -> Tensor:
        """Compute total KL divergence for all variational layers."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.hidden_layers:
            kl += layer.kl_divergence()
        kl += self.output_layer.kl_divergence()
        if self.learn_noise:
            kl += self.noise_layer.kl_divergence()
        return kl

    def elbo_loss(
        self,
        x: Tensor,
        y: Tensor,
        n_samples: int = 1,
        kl_weight: float = 1.0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute the Evidence Lower Bound (ELBO) loss.

        ELBO = E_q[log p(y|x,w)] - KL(q(w)||p(w))

        Args:
            x: Input tensor
            y: Target tensor
            n_samples: Number of MC samples for expectation
            kl_weight: Weight for KL term (for KL annealing)

        Returns:
            loss: Negative ELBO (to minimize)
            nll: Negative log-likelihood term
            kl: KL divergence term
        """
        nll = torch.tensor(0.0, device=x.device)

        for _ in range(n_samples):
            mean, log_var = self.forward(x, sample=True)

            if self.learn_noise:
                # Heteroscedastic Gaussian likelihood
                var = torch.exp(log_var) + 1e-6
                nll += 0.5 * torch.mean(
                    log_var + (y - mean) ** 2 / var
                )
            else:
                # Homoscedastic MSE (assumes unit variance)
                nll += 0.5 * torch.mean((y - mean) ** 2)

        nll = nll / n_samples
        kl = self.kl_divergence() / x.shape[0]  # Normalize by batch size

        loss = nll + kl_weight * kl
        return loss, nll, kl

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        n_samples: int = 100,
        return_individual: bool = False
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Make predictions with uncertainty estimates.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            n_samples: Number of MC samples
            return_individual: If True, also return individual samples

        Returns:
            mean: Mean prediction
            std: Total uncertainty (epistemic + aleatoric)
            samples: Individual predictions (if return_individual=True)
        """
        self.eval()

        means = []
        log_vars = []

        for _ in range(n_samples):
            mean, log_var = self.forward(x, sample=True)
            means.append(mean)
            if log_var is not None:
                log_vars.append(log_var)

        means = torch.stack(means, dim=0)  # (n_samples, batch_size, output_dim)

        # Epistemic uncertainty: variance of means
        epistemic_var = torch.var(means, dim=0)

        # Mean prediction
        pred_mean = torch.mean(means, dim=0)

        if self.learn_noise and log_vars:
            # Aleatoric uncertainty: mean of variances
            log_vars = torch.stack(log_vars, dim=0)
            aleatoric_var = torch.mean(torch.exp(log_vars), dim=0)
            total_var = epistemic_var + aleatoric_var
        else:
            total_var = epistemic_var

        total_std = torch.sqrt(total_var + 1e-6)

        if return_individual:
            return pred_mean, total_std, means
        else:
            return pred_mean, total_std, None

    @torch.no_grad()
    def predict_decomposed(
        self,
        x: Tensor,
        n_samples: int = 100
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Predict with decomposed uncertainty estimates.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            mean: Mean prediction
            epistemic_std: Epistemic uncertainty
            aleatoric_std: Aleatoric uncertainty (zeros if learn_noise=False)
        """
        self.eval()

        means = []
        log_vars = []

        for _ in range(n_samples):
            mean, log_var = self.forward(x, sample=True)
            means.append(mean)
            if log_var is not None:
                log_vars.append(log_var)

        means = torch.stack(means, dim=0)
        pred_mean = torch.mean(means, dim=0)
        epistemic_std = torch.std(means, dim=0)

        if self.learn_noise and log_vars:
            log_vars = torch.stack(log_vars, dim=0)
            aleatoric_std = torch.sqrt(torch.mean(torch.exp(log_vars), dim=0) + 1e-6)
        else:
            aleatoric_std = torch.zeros_like(pred_mean)

        return pred_mean, epistemic_std, aleatoric_std


class MCDropoutNN(nn.Module):
    """
    Neural Network with MC Dropout for uncertainty estimation.

    Uses dropout at test time to approximate Bayesian inference.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: List[int] = [50, 50],
        output_dim: int = 1,
        dropout_rate: float = 0.1,
        learn_noise: bool = True,
        activation: str = "tanh"
    ):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Dimension of output
            dropout_rate: Dropout probability
            learn_noise: If True, learn heteroscedastic noise
            activation: Activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.learn_noise = learn_noise

        # Build network
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)

        # Output heads
        self.mean_head = nn.Linear(prev_dim, output_dim)
        if learn_noise:
            self.log_var_head = nn.Linear(prev_dim, output_dim)

        # Activation
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        self.activation = activations.get(activation, nn.Tanh())

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass (dropout always active for MC estimation)."""
        h = x
        for i, layer in enumerate(self.hidden_layers):
            if isinstance(layer, nn.Linear):
                h = self.activation(layer(h))
            else:  # Dropout
                h = layer(h)

        mean = self.mean_head(h)

        if self.learn_noise:
            log_var = self.log_var_head(h)
            return mean, log_var
        else:
            return mean, None

    def loss(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute loss (NLL for heteroscedastic, MSE otherwise)."""
        mean, log_var = self.forward(x)

        if self.learn_noise:
            var = torch.exp(log_var) + 1e-6
            loss = 0.5 * torch.mean(log_var + (y - mean) ** 2 / var)
        else:
            loss = 0.5 * torch.mean((y - mean) ** 2)

        return loss

    def enable_dropout(self) -> None:
        """Enable dropout for MC sampling (call before predict)."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        n_samples: int = 100
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Make predictions with MC Dropout uncertainty.

        Args:
            x: Input tensor
            n_samples: Number of forward passes

        Returns:
            mean: Mean prediction
            std: Total uncertainty
            samples: Individual samples (None for this implementation)
        """
        self.enable_dropout()

        means = []
        log_vars = []

        for _ in range(n_samples):
            mean, log_var = self.forward(x)
            means.append(mean)
            if log_var is not None:
                log_vars.append(log_var)

        means = torch.stack(means, dim=0)
        pred_mean = torch.mean(means, dim=0)
        epistemic_var = torch.var(means, dim=0)

        if self.learn_noise and log_vars:
            log_vars = torch.stack(log_vars, dim=0)
            aleatoric_var = torch.mean(torch.exp(log_vars), dim=0)
            total_var = epistemic_var + aleatoric_var
        else:
            total_var = epistemic_var

        total_std = torch.sqrt(total_var + 1e-6)

        return pred_mean, total_std, None

    @torch.no_grad()
    def predict_decomposed(
        self,
        x: Tensor,
        n_samples: int = 100
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict with decomposed uncertainty estimates."""
        self.enable_dropout()

        means = []
        log_vars = []

        for _ in range(n_samples):
            mean, log_var = self.forward(x)
            means.append(mean)
            if log_var is not None:
                log_vars.append(log_var)

        means = torch.stack(means, dim=0)
        pred_mean = torch.mean(means, dim=0)
        epistemic_std = torch.std(means, dim=0)

        if self.learn_noise and log_vars:
            log_vars = torch.stack(log_vars, dim=0)
            aleatoric_std = torch.sqrt(torch.mean(torch.exp(log_vars), dim=0) + 1e-6)
        else:
            aleatoric_std = torch.zeros_like(pred_mean)

        return pred_mean, epistemic_std, aleatoric_std

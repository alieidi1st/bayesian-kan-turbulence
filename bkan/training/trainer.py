"""
Training utilities for Bayesian Neural Networks.
"""

from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer, Adam
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR

from ..models.bayesian_nn import BayesianNN, MCDropoutNN


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 1000
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    kl_annealing_epochs: int = 100
    n_mc_samples: int = 1
    early_stopping_patience: int = 50
    scheduler: str = "cosine"  # "cosine", "none"
    verbose: bool = True
    print_every: int = 100


@dataclass
class TrainingHistory:
    """Training history container."""
    loss: List[float] = field(default_factory=list)
    nll: List[float] = field(default_factory=list)
    kl: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)


class BNNTrainer:
    """
    Trainer for Bayesian Neural Networks.

    Handles:
    - ELBO optimization with KL annealing
    - Learning rate scheduling
    - Early stopping
    - Training history logging
    """

    def __init__(
        self,
        model: BayesianNN,
        config: Optional[TrainingConfig] = None,
        device: str = "cpu"
    ):
        """
        Args:
            model: BayesianNN model to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config or TrainingConfig()
        self.device = device
        self.history = TrainingHistory()

    def _create_dataloader(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        shuffle: bool = True
    ) -> DataLoader:
        """Create a DataLoader from tensors."""
        dataset = TensorDataset(x.to(self.device), y.to(self.device))
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle
        )

    def _get_kl_weight(self, epoch: int) -> float:
        """Compute KL weight for annealing."""
        if self.config.kl_annealing_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.config.kl_annealing_epochs)

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None
    ) -> TrainingHistory:
        """
        Train the model.

        Args:
            x_train: Training inputs
            y_train: Training targets
            x_val: Validation inputs (optional)
            y_val: Validation targets (optional)

        Returns:
            Training history
        """
        train_loader = self._create_dataloader(x_train, y_train, shuffle=True)
        dataset_size = x_train.shape[0]

        optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = None
        if self.config.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_nll = 0.0
            epoch_kl = 0.0
            n_batches = 0

            kl_weight = self._get_kl_weight(epoch)

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()

                loss, nll, kl = self.model.elbo_loss(
                    x_batch, y_batch,
                    n_samples=self.config.n_mc_samples,
                    kl_weight=kl_weight,
                    dataset_size=dataset_size,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_nll += nll.item()
                epoch_kl += kl.item()
                n_batches += 1

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / n_batches
            avg_nll = epoch_nll / n_batches
            avg_kl = epoch_kl / n_batches

            self.history.loss.append(avg_loss)
            self.history.nll.append(avg_nll)
            self.history.kl.append(avg_kl)

            # Validation
            if x_val is not None and y_val is not None:
                val_loss = self._validate(x_val, y_val)
                self.history.val_loss.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        if self.config.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Logging
            if self.config.verbose and (epoch + 1) % self.config.print_every == 0:
                msg = f"Epoch {epoch + 1}/{self.config.epochs}: "
                msg += f"Loss={avg_loss:.4f}, NLL={avg_nll:.4f}, KL={avg_kl:.4f}"
                if x_val is not None:
                    msg += f", Val={self.history.val_loss[-1]:.4f}"
                print(msg)

        return self.history

    def _validate(self, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
        """Compute validation loss."""
        self.model.eval()
        with torch.no_grad():
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)
            loss, _, _ = self.model.elbo_loss(x_val, y_val, n_samples=10)
        return loss.item()


class MCDropoutTrainer:
    """
    Trainer for MC Dropout Neural Networks.
    """

    def __init__(
        self,
        model: MCDropoutNN,
        config: Optional[TrainingConfig] = None,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.config = config or TrainingConfig()
        self.device = device
        self.history = TrainingHistory()

    def _create_dataloader(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        shuffle: bool = True
    ) -> DataLoader:
        dataset = TensorDataset(x.to(self.device), y.to(self.device))
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle
        )

    def train(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None
    ) -> TrainingHistory:
        """Train the MC Dropout model."""
        train_loader = self._create_dataloader(x_train, y_train, shuffle=True)

        optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        scheduler = None
        if self.config.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = self.model.loss(x_batch, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / n_batches
            self.history.loss.append(avg_loss)

            # Validation
            if x_val is not None and y_val is not None:
                val_loss = self._validate(x_val, y_val)
                self.history.val_loss.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        if self.config.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

            if self.config.verbose and (epoch + 1) % self.config.print_every == 0:
                msg = f"Epoch {epoch + 1}/{self.config.epochs}: Loss={avg_loss:.4f}"
                if x_val is not None:
                    msg += f", Val={self.history.val_loss[-1]:.4f}"
                print(msg)

        return self.history

    def _validate(self, x_val: torch.Tensor, y_val: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            x_val = x_val.to(self.device)
            y_val = y_val.to(self.device)
            loss = self.model.loss(x_val, y_val)
        return loss.item()

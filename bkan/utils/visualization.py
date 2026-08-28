"""
Visualization utilities for Bayesian Neural Network predictions.
"""

from typing import Optional, Tuple, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_predictions(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    y_true: Optional[np.ndarray] = None,
    title: str = "BNN Predictions",
    xlabel: str = "x",
    ylabel: str = "y",
    confidence_intervals: List[float] = [1, 2, 3],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot predictions with uncertainty bands.

    Args:
        x_train: Training inputs
        y_train: Training targets
        x_test: Test inputs
        pred_mean: Predicted mean
        pred_std: Predicted standard deviation
        y_true: True function values (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        confidence_intervals: List of sigma values for uncertainty bands
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy
    x_train_np = x_train.cpu().numpy().flatten()
    y_train_np = y_train.cpu().numpy().flatten()
    x_test_np = x_test.cpu().numpy().flatten()
    mean_np = pred_mean.cpu().numpy().flatten()
    std_np = pred_std.cpu().numpy().flatten()

    # Sort by x for proper line plotting
    sort_idx = np.argsort(x_test_np)
    x_test_np = x_test_np[sort_idx]
    mean_np = mean_np[sort_idx]
    std_np = std_np[sort_idx]

    # Plot uncertainty bands (from outermost to innermost)
    colors = plt.cm.Blues(np.linspace(0.2, 0.5, len(confidence_intervals)))
    for i, sigma in enumerate(sorted(confidence_intervals, reverse=True)):
        ax.fill_between(
            x_test_np,
            mean_np - sigma * std_np,
            mean_np + sigma * std_np,
            alpha=0.3,
            color=colors[i],
            label=f"±{sigma}σ"
        )

    # Plot true function if provided
    if y_true is not None:
        if len(y_true) == len(x_test_np):
            ax.plot(x_test_np, y_true[sort_idx], 'g--', linewidth=2, label='True')
        else:
            ax.plot(x_test_np, y_true, 'g--', linewidth=2, label='True')

    # Plot mean prediction
    ax.plot(x_test_np, mean_np, 'b-', linewidth=2, label='Mean')

    # Plot training data
    ax.scatter(x_train_np, y_train_np, c='red', s=10, alpha=0.5, label='Train', zorder=5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_decomposed_uncertainty(
    x_test: torch.Tensor,
    pred_mean: torch.Tensor,
    epistemic_std: torch.Tensor,
    aleatoric_std: torch.Tensor,
    x_train: Optional[torch.Tensor] = None,
    y_train: Optional[torch.Tensor] = None,
    title: str = "Decomposed Uncertainty",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot predictions with decomposed epistemic and aleatoric uncertainty.

    Args:
        x_test: Test inputs
        pred_mean: Predicted mean
        epistemic_std: Epistemic uncertainty
        aleatoric_std: Aleatoric uncertainty
        x_train: Training inputs (optional)
        y_train: Training targets (optional)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Convert to numpy
    x_np = x_test.cpu().numpy().flatten()
    mean_np = pred_mean.cpu().numpy().flatten()
    epi_np = epistemic_std.cpu().numpy().flatten()
    ale_np = aleatoric_std.cpu().numpy().flatten()

    sort_idx = np.argsort(x_np)
    x_np = x_np[sort_idx]
    mean_np = mean_np[sort_idx]
    epi_np = epi_np[sort_idx]
    ale_np = ale_np[sort_idx]

    total_std = np.sqrt(epi_np**2 + ale_np**2)

    # Top plot: Predictions with total uncertainty
    ax = axes[0]
    ax.fill_between(x_np, mean_np - 2*total_std, mean_np + 2*total_std,
                    alpha=0.3, color='blue', label='±2σ total')
    ax.plot(x_np, mean_np, 'b-', linewidth=2, label='Mean')

    if x_train is not None and y_train is not None:
        ax.scatter(x_train.cpu().numpy().flatten(),
                   y_train.cpu().numpy().flatten(),
                   c='red', s=10, alpha=0.5, label='Train')

    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Bottom plot: Decomposed uncertainty
    ax = axes[1]
    ax.fill_between(x_np, 0, epi_np, alpha=0.5, color='orange', label='Epistemic')
    ax.fill_between(x_np, epi_np, epi_np + ale_np, alpha=0.5, color='purple', label='Aleatoric')
    ax.plot(x_np, total_std, 'k-', linewidth=2, label='Total')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Uncertainty (σ)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_training_history(
    history,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot training history (loss, NLL, KL).

    Args:
        history: TrainingHistory object
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    n_plots = 1
    if hasattr(history, 'nll') and len(history.nll) > 0:
        n_plots = 3

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Loss
    axes[0].plot(history.loss, label='Train')
    if len(history.val_loss) > 0:
        axes[0].plot(history.val_loss, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if n_plots == 3:
        # NLL
        axes[1].plot(history.nll)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('NLL')
        axes[1].set_title('Negative Log-Likelihood')
        axes[1].grid(True, alpha=0.3)

        # KL
        axes[2].plot(history.kl)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('KL')
        axes[2].set_title('KL Divergence')
        axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_all_toy_problems(
    results: dict,
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None
) -> Figure:
    """
    Plot results for all toy problems in a grid.

    Args:
        results: Dictionary with problem names as keys, containing:
            - x_train, y_train, x_test
            - pred_mean, pred_std
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    n_problems = len(results)
    n_cols = 3
    n_rows = (n_problems + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, (name, data) in enumerate(results.items()):
        ax = axes[i]

        x_train = data['x_train'].cpu().numpy().flatten()
        y_train = data['y_train'].cpu().numpy().flatten()
        x_test = data['x_test'].cpu().numpy().flatten()
        mean = data['pred_mean'].cpu().numpy().flatten()
        std = data['pred_std'].cpu().numpy().flatten()

        sort_idx = np.argsort(x_test)
        x_test = x_test[sort_idx]
        mean = mean[sort_idx]
        std = std[sort_idx]

        ax.fill_between(x_test, mean - 2*std, mean + 2*std,
                        alpha=0.3, color='blue')
        ax.plot(x_test, mean, 'b-', linewidth=2)
        ax.scatter(x_train, y_train, c='red', s=5, alpha=0.3)

        ax.set_title(name.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig

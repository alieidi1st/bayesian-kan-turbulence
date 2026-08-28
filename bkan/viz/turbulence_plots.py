"""
Field plotting on case geometry for the a-priori turbulence study.

The curated dataset provides cell-centre coordinates (C_1, C_2, C_3) for every
point, so any per-cell quantity (DNS anisotropy, BKAN prediction, error, or
predictive uncertainty) can be drawn as a field over the actual flow geometry,
exactly as in the dataset's example_code.py. Planar flows (flat plate, periodic
hills, converging-diverging channel, curved backward-facing step) live in the
x1-x2 plane; the square duct cross-section lives in the x2-x3 plane.
"""

import numpy as np
import matplotlib.pyplot as plt


def case_plane(case: str) -> tuple:
    """Return the (i, j) coordinate-column indices to plot for a given case."""
    return (1, 2) if 'squareDuct' in case else (0, 1)


def plot_field(ax, coords, values, plane=(0, 1), cmap='RdBu_r',
               vmin=None, vmax=None, s=4, title=None, symmetric=False):
    """
    Scatter a per-cell field over the case geometry.

    Parameters
    ----------
    ax : matplotlib Axes
    coords : (n,3) cell-centre coordinates
    values : (n,) field to colour by
    plane : (i,j) coordinate columns to use as plot axes
    symmetric : if True and vmin/vmax are None, use a symmetric colour range.
    """
    i, j = plane
    if vmin is None and vmax is None:
        if symmetric:
            m = np.percentile(np.abs(values), 99)
            vmin, vmax = -m, m
        else:
            vmin, vmax = np.percentile(values, [1, 99])
    sc = ax.scatter(coords[:, i], coords[:, j], c=values, cmap=cmap,
                    vmin=vmin, vmax=vmax, s=s, linewidths=0, rasterized=True)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return sc


def plot_component_summary(coords, b_true, b_pred, epi_std, comp=1, case='',
                           comp_names=('b11', 'b12', 'b13', 'b22', 'b23', 'b33'),
                           figsize=(11, 3)):
    """
    Four-panel field summary for one anisotropy component over the geometry:
    DNS reference, BKAN mean prediction, absolute error, and epistemic std.

    b_true, b_pred, epi_std : (n,6) arrays (unique components 11,12,13,22,23,33).

    Returns the matplotlib Figure.
    """
    plane = case_plane(case)
    name = comp_names[comp]
    err = np.abs(b_pred[:, comp] - b_true[:, comp])

    m = np.percentile(np.abs(b_true[:, comp]), 99)
    fig, axes = plt.subplots(1, 4, figsize=figsize, constrained_layout=True)
    sc0 = plot_field(axes[0], coords, b_true[:, comp], plane, vmin=-m, vmax=m,
                     title=f'DNS {name}')
    plot_field(axes[1], coords, b_pred[:, comp], plane, vmin=-m, vmax=m,
               title=f'BKAN {name}')
    sc2 = plot_field(axes[2], coords, err, plane, cmap='viridis',
                     vmin=0, vmax=np.percentile(err, 99), title='|error|')
    sc3 = plot_field(axes[3], coords, epi_std[:, comp], plane, cmap='magma',
                     vmin=0, vmax=np.percentile(epi_std[:, comp], 99),
                     title='epistemic std')
    fig.colorbar(sc0, ax=axes[:2], shrink=0.8, location='bottom', pad=0.02)
    fig.colorbar(sc2, ax=[axes[2]], shrink=0.8, location='bottom', pad=0.02)
    fig.colorbar(sc3, ax=[axes[3]], shrink=0.8, location='bottom', pad=0.02)
    fig.suptitle(f'{case}  —  anisotropy component {name}', fontsize=11)
    return fig


def plot_training_curves(history, path, title=''):
    """
    Four-panel training-convergence figure (ELBO loss, NLL, realizability, KL/N)
    versus epoch, in the style of standard Bayesian-NN reports.

    history : list of dicts with keys 'loss','nll','realiz','kl' (per epoch).
    """
    ep = np.arange(len(history))
    keys = [('loss', 'ELBO loss', 'C0'), ('nll', 'negative log-likelihood', 'C1'),
            ('realiz', 'realizability penalty', 'C2'), ('kl', 'KL / N', 'C3')]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for a, (k, lab, col) in zip(ax.flat, keys):
        y = np.array([h[k] for h in history])
        a.plot(ep, y, color=col, lw=1.2)
        a.set_xlabel('epoch'); a.set_ylabel(lab)
        if k == 'realiz' and (y > 0).any():
            a.set_yscale('log')
        a.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_uncertainty_decomposition(coords, epi, alea, comp, case, path,
                                   comp_names=('b11', 'b12', 'b13', 'b22', 'b23', 'b33')):
    """Three-panel epistemic / aleatoric / total uncertainty fields over geometry."""
    plane = case_plane(case)
    name = comp_names[comp]
    tot = np.sqrt(epi[:, comp] ** 2 + alea[:, comp] ** 2)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), constrained_layout=True)
    for a, (val, lab, cm) in zip(axes, [
            (epi[:, comp], 'epistemic', 'magma'),
            (alea[:, comp], 'aleatoric', 'cividis'),
            (tot, 'total', 'viridis')]):
        sc = plot_field(a, coords, val, plane, cmap=cm, vmin=0,
                        vmax=np.percentile(val, 99), title=f'{lab} std')
        fig.colorbar(sc, ax=a, shrink=0.8, location='bottom', pad=0.02)
    fig.suptitle(f'{case} — uncertainty decomposition of {name}', fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_all_components(coords, b_true, b_pred, epi, case, path,
                        comps=(0, 1, 3, 5),
                        comp_names=('b11', 'b12', 'b13', 'b22', 'b23', 'b33')):
    """
    Grid of anisotropy components (columns) vs. rows {DNS, BKAN mean, epistemic}
    as fields over the case geometry.
    """
    plane = case_plane(case)
    ncol = len(comps)
    fig, axes = plt.subplots(3, ncol, figsize=(3.0 * ncol, 7),
                             constrained_layout=True)
    for c, comp in enumerate(comps):
        m = np.percentile(np.abs(b_true[:, comp]), 99) + 1e-9
        s0 = plot_field(axes[0, c], coords, b_true[:, comp], plane, vmin=-m, vmax=m,
                        title=comp_names[comp])
        plot_field(axes[1, c], coords, b_pred[:, comp], plane, vmin=-m, vmax=m)
        ev = epi[:, comp]
        s2 = plot_field(axes[2, c], coords, ev, plane, cmap='magma', vmin=0,
                        vmax=np.percentile(ev, 99) + 1e-9)
        if c == 0:
            axes[0, c].set_ylabel('DNS', fontsize=9)
            axes[1, c].set_ylabel('BKAN mean', fontsize=9)
            axes[2, c].set_ylabel('epistemic', fontsize=9)
    fig.suptitle(f'{case} — anisotropy components', fontsize=11)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_coverage_bars(cov1, cov2, path, case='',
                       comp_names=('b11', 'b12', 'b13', 'b22', 'b23', 'b33')):
    """Bar chart of 1-sigma / 2-sigma empirical coverage per component vs. ideal."""
    x = np.arange(len(comp_names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - w / 2, 100 * np.asarray(cov1), w, label='within 1σ', color='#5b9bd5')
    ax.bar(x + w / 2, 100 * np.asarray(cov2), w, label='within 2σ', color='#c0504d')
    ax.axhline(68, ls=':', c='gray', lw=1); ax.axhline(95, ls='--', c='gray', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(comp_names)
    ax.set_ylabel('coverage [%]'); ax.set_ylim(0, 105)
    ax.set_title(f'{case} — predictive-interval coverage'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_barycentric(ax, xy, c=None, cmap='viridis', s=4, title=None):
    """
    Plot anisotropy states on the barycentric (Lumley) triangle.

    xy : (n,2) barycentric coordinates from tensor_basis.barycentric_coords.
    c  : optional per-point colour (e.g. epistemic uncertainty).
    """
    # triangle edges: corners 1c=(1,0), 2c=(0,0), 3c=(0.5, sqrt3/2)
    corners = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2], [1, 0]])
    ax.plot(corners[:, 0], corners[:, 1], 'k-', lw=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=s, linewidths=0,
                    rasterized=True)
    ax.text(1, -0.03, '1-comp', ha='center', va='top', fontsize=8)
    ax.text(0, -0.03, '2-comp', ha='center', va='top', fontsize=8)
    ax.text(0.5, np.sqrt(3) / 2 + 0.02, '3-comp (iso)', ha='center', fontsize=8)
    ax.set_aspect('equal'); ax.axis('off')
    if title:
        ax.set_title(title, fontsize=10)
    return sc

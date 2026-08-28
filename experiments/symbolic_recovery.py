"""
Uncertainty-aware symbolic recovery of the tensor-basis coefficient functions.

For interpretability we train the MINIMAL two-invariant tensor-basis BKAN (inputs
lambda_1, lambda_2; no extra features) on a 2-D split, then distil each learned
coefficient function g_n into a closed form WITH confidence bounds: we sweep the
dominant invariant lambda_1 (holding lambda_2 at its median), read off the
posterior mean and +/- 2 sigma band of g_n, and fit c f(a x + b) + d from the
symbolic library, scoring the fit against the mean and against the bounds.

Outputs a figure (g_n vs lambda_1 with posterior band and symbolic fit) and a
JSON table of the recovered closed forms.

Usage
-----
    python experiments/symbolic_recovery.py --split periodic_hills
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bkan.data.turbulence import SPLITS, load_split
from bkan.data import tensor_basis as tb
from bkan.models.tbkan import TensorBasisBKAN
from bkan.models.symbolic_utils import suggest_symbolic
from scripts.turbulence_apriori import train

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(REPO, 'reports', 'figures', 'turbulence')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='periodic_hills')
    ap.add_argument('--hidden', type=int, nargs='+', default=[16])
    ap.add_argument('--pretrain', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results',
                                                  'symbolic_results.json'))
    ap.add_argument('--seed', type=int, default=0,
                    help='seed for model init + posterior sampling (reproducible '
                         'symbolic recovery; the decomposition is otherwise '
                         'sensitive to the random initialisation)')
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cfg = SPLITS[args.split]
    assert cfg['dim'] == 2, 'symbolic recovery uses a 2-D, 2-invariant split'
    train_d, _, _ = load_split(args.split)
    lam = tb.compute_invariants(train_d['S'], train_d['R'], train_d['k'],
                                train_d['eps'], n_inv=2)
    z, stats = tb.standardize(lam)
    z = np.clip(z, -4, 4)
    T = tb.normalize_basis(tb.compute_basis(train_d['S'], train_d['R'],
                                            train_d['k'], train_d['eps'], n_basis=3))
    X = torch.tensor(z, dtype=torch.float32)
    Tt = torch.tensor(T, dtype=torch.float32)
    b = torch.tensor(train_d['b'], dtype=torch.float32)

    # minimal two-invariant model (no extra features) for interpretability
    model = TensorBasisBKAN(n_inv=2, hidden_dims=args.hidden, n_basis=3,
                            noise='heteroscedastic')
    train(model, X, Tt, b, args.pretrain, 8192, 5e-3, 100, bayesian=False)
    train(model, X, Tt, b, args.epochs, 8192, 5e-3, 100, bayesian=True)

    # sweep standardised lambda_1 at median lambda_2; posterior mean/std of g_n
    grid = np.linspace(-2.5, 2.5, 200).astype(np.float32)
    xg = torch.tensor(np.stack([grid, np.zeros_like(grid)], axis=1))
    g_mean, g_std = model.predict_coefficients(xg, n_samples=200)
    g_mean, g_std = g_mean.numpy(), g_std.numpy()
    # data-dense range of the standardised invariant (where the fit is supported,
    # as opposed to extrapolated); shaded on each panel
    lo, hi = np.percentile(z[:, 0], [2, 98])

    from bkan.models.symbolic_utils import SYMBOLIC_LIB

    def _n(v, signed=True):
        """Format a coefficient, using compact sci-notation for tiny/large values."""
        s = f'{v:+.2g}' if (v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e3)) else f'{v:+.2f}'
        return s if signed else s.lstrip('+')

    def pretty(cand):
        """Compact mathtext of the fitted closed form c*f(a*lam1+b)+d."""
        a, b, c, d, name = (cand['a'], cand['b'], cand['c'], cand['d'], cand['name'])
        u = f'{_n(a, signed=False)}\\lambda_1{_n(b)}'
        if name == 'tanh(x)':
            core = f'\\tanh({u})'
        elif name == 'x*exp(-x^2)':
            core = f'({u})\\,e^{{-({u})^2}}'
        elif name == 'x^2':
            core = f'({u})^2'
        elif name == 'x^4':
            core = f'({u})^4'
        else:                                   # generic fallback
            core = name.replace('x', f'({u})')
        return f'${_n(c)}\\,{core}{_n(d)}$'

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
    table = []
    for n in range(3):
        ym, ys = g_mean[:, n], g_std[:, n]
        cand = suggest_symbolic(grid, ym, ys, n_top=3)[0]
        table.append({'coefficient': f'g{n+1}', 'formula': cand['formula'],
                      'expression': pretty(cand).strip('$'),
                      'r2_mean': cand['r2_mean'], 'r2_lower': cand['r2_lower'],
                      'r2_upper': cand['r2_upper'], 'name': cand['name']})
        ax[n].axvspan(lo, hi, color='0.88', zorder=0,
                      label='data range (2--98\\%)' if n == 0 else None)
        ax[n].fill_between(grid, ym - 2 * ys, ym + 2 * ys, color='C0', alpha=0.25,
                           lw=0, zorder=2, label=r'posterior $\pm2\sigma$')
        ax[n].plot(grid, ym, color='C0', lw=2.0, zorder=3, label='posterior mean')
        # fitted symbolic form -- drawn last, thick high-contrast dashes so it
        # stays visible when the figure is printed at column width
        f = SYMBOLIC_LIB[cand['name']]
        yfit = cand['c'] * f(cand['a'] * grid + cand['b']) + cand['d']
        ax[n].plot(grid, yfit, color='crimson', lw=2.4, ls=(0, (6, 3)), zorder=5,
                   label='symbolic fit')
        good = cand['r2_mean'] >= 0.9
        ax[n].set_title(f"$g_{n+1}(\\lambda_1)$   "
                        f"$R^2={cand['r2_mean']:.2f}$"
                        + ('' if good else '  (no simple closed form)'),
                        fontsize=10)
        # the recovered expression, annotated inside the panel
        ax[n].text(0.5, 0.04, pretty(cand), transform=ax[n].transAxes,
                   ha='center', va='bottom', fontsize=8.5,
                   color='crimson' if good else '0.4',
                   bbox=dict(boxstyle='round,pad=0.25', fc='white',
                             ec='0.7', lw=0.5, alpha=0.85))
        ax[n].set_xlabel(r'standardised invariant $\lambda_1$')
        ax[n].set_xlim(-2.5, 2.5); ax[n].grid(alpha=0.3)
        if n == 0:
            ax[n].legend(fontsize=8, loc='upper right')
            ax[n].set_ylabel(r'tensor-basis coefficient $g_n$')
    fig.suptitle(f'{cfg["label"]} — uncertainty-aware symbolic recovery of the '
                 r'coefficient functions $g_n(\lambda_1)$', fontsize=11)
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, f'{args.split}_symbolic.png'), dpi=200)
    # write straight into the paper's figure directory (png is what JCP.tex uses)
    jcpfig = os.path.expanduser('~/Desktop/KAN/JCP_paper/figures')
    if os.path.isdir(jcpfig):
        fig.savefig(os.path.join(jcpfig, 'fig02_edge_interpretability.png'), dpi=200)
        fig.savefig(os.path.join(jcpfig, 'fig02_edge_interpretability.pdf'))
        print('  updated fig02_edge_interpretability.{png,pdf} in', jcpfig)
    plt.close(fig)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'split': args.split, 'coefficients': table}, f, indent=2)
    for r in table:
        print(f"{r['coefficient']}: {r['formula']}  (R2={r['r2_mean']:.3f}, "
              f"bounds {r['r2_lower']:.3f}/{r['r2_upper']:.3f})", flush=True)
    print('wrote', args.out, 'and figure', flush=True)


if __name__ == '__main__':
    main()

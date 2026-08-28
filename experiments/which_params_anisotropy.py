"""
Systematic 'which KAN parameters to make Bayesian' ablation, on the turbulence
anisotropy split (periodic hills). Compares the three variants -- spline
coefficients (b-coef), edge amplitudes (b-scale), and both (b-full) -- reporting
accuracy (RMSE), calibration error, and the mean epistemic / aleatoric
uncertainty. This makes the ablation systematic on a real dataset (vs. the single
1-D toy in the report).
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bkan.data.turbulence import load_split, SPLITS
from bkan.models.tbkan import TensorBasisBKAN
from scripts.turbulence_apriori import prep, train, evaluate, n_features


def run(split='periodic_hills', variants=('coef', 'scale', 'full')):
    cfg = SPLITS[split]
    n_inv, n_basis = cfg['n_inv'], cfg['n_basis']
    nfeat = n_features(n_inv, extra=True)
    tr, te, _ = load_split(split)
    Xtr, Ttr, btr, stats = prep(tr, n_inv=n_inv, n_basis=n_basis, extra=True)
    Xte, Tte, bte, _ = prep(te, stats, n_inv=n_inv, n_basis=n_basis, extra=True)
    rows = []
    for v in variants:
        m = TensorBasisBKAN(n_inv=nfeat, hidden_dims=[16], n_basis=n_basis,
                            realiz_weight=10.0, noise='heteroscedastic', variant=v)
        train(m, Xtr, Ttr, btr, 60, 8192, 5e-3, 100, bayesian=False)
        train(m, Xtr, Ttr, btr, 250, 8192, 5e-3, 100, bayesian=True)
        met, *_ = evaluate(m, Xte, Tte, bte, bayesian=True, n_samples=50)
        n_param = sum(p.numel() for p in m.parameters() if p.requires_grad)
        row = {'variant': v, 'params': int(n_param), 'rmse': met['rmse'],
               'calibration_error': met['calibration_error'],
               'epistemic_mean': met['epistemic_mean'],
               'aleatoric_mean': met['aleatoric_mean']}
        rows.append(row)
        print(f"  b-{v:5s}: params={n_param:5d} RMSE={row['rmse']:.4f} "
              f"cal={row['calibration_error']:.3f} epi={row['epistemic_mean']:.4f} "
              f"alea={row['aleatoric_mean']:.4f}", flush=True)
    return rows


if __name__ == '__main__':
    print('=== which-params ablation: turbulence (periodic hills) ===', flush=True)
    rows = run('periodic_hills')
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'which_params_anisotropy.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'split': 'periodic_hills', 'rows': rows}, open(out, 'w'), indent=2)
    print('wrote', out, flush=True)

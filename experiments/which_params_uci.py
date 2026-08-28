"""
'Which KAN parameters to make Bayesian' ablation on the UCI regression benchmark,
the companion to experiments/which_params_anisotropy.py (turbulence). For each dataset we train
the BKAN at the matched HL width with the three posterior variants -- spline
coefficients (b-coef), edge amplitudes (b-scale) and both (b-full) -- under the
Hernandez-Lobato protocol, and report multi-split RMSE / test log-likelihood /
calibration error plus the mean epistemic uncertainty. Together with the
turbulence ablation this makes the 'which-params' study systematic across both a
controlled benchmark and the target application.

Results -> results/which_params_uci.csv.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bkan.data.uci import load_uci, n_splits, HL_HIDDEN_UNITS
from bkan.models import BNN, BayesianKAN
from scripts.uci_benchmark import (n_params, matched_kan_width, evaluate,
                                   train_model, epochs_for)

VARIANTS = ('coef', 'scale', 'full')
# small/medium datasets -- the ablation is about the posterior, not scale
DEFAULT_DATASETS = ['yacht', 'boston', 'energy', 'concrete', 'wine']


@torch.no_grad()
def epistemic_mean(model, x_te, y_std, n_mc=100):
    """Mean epistemic std over the test set, in original target units."""
    _, epi, _ = model.predict_decomposed(x_te, n_samples=n_mc)
    return float(epi.cpu().numpy().mean() * y_std)


def run_dataset(name, variants, cap_splits, n_mc, device):
    ns = n_splits(name)
    if cap_splits is not None:
        ns = min(ns, cap_splits)
    hl = HL_HIDDEN_UNITS[name]
    d0 = load_uci(name, 0)
    ref_params = n_params(BNN(d0['input_dim'], [hl], 1, prior_type='gaussian',
                              noise='homoscedastic'))
    kan_width = matched_kan_width(d0['input_dim'], ref_params)
    epochs = epochs_for(len(d0['x_train']))

    acc = {v: {'rmse': [], 'll': [], 'ece': [], 'epi': [], 'params': 0} for v in variants}
    for split in range(ns):
        d = load_uci(name, split)
        x_tr = torch.tensor(d['x_train']); y_tr = torch.tensor(d['y_train'])
        x_te = torch.tensor(d['x_test']).to(device); y_te = torch.tensor(d['y_test'])
        for v in variants:
            torch.manual_seed(split)
            m = BayesianKAN(d0['input_dim'], [kan_width], 1, variant=v,
                            prior_type='gaussian', noise='homoscedastic').to(device)
            train_model(m, x_tr, y_tr, 'elbo', epochs, device)
            rmse, ll, ece = evaluate(m, x_te, y_te, d['y_std'], n_mc)
            acc[v]['rmse'].append(rmse); acc[v]['ll'].append(ll); acc[v]['ece'].append(ece)
            acc[v]['epi'].append(epistemic_mean(m, x_te, d['y_std'], n_mc))
            acc[v]['params'] = n_params(m)

    print(f'=== {name} (dim={d0["input_dim"]}, N={len(d0["x_train"])}, splits={ns}, '
          f'KAN width={kan_width}, epochs={epochs}) ===', flush=True)
    rows = []
    for v in variants:
        r = np.array(acc[v]['rmse']); l = np.array(acc[v]['ll'])
        e = np.array(acc[v]['ece']); ep = np.array(acc[v]['epi'])
        print(f'  b-{v:5s} p={acc[v]["params"]:5d}  RMSE={r.mean():.3f}+/-{r.std():.3f}'
              f'  LL={l.mean():.3f}  ECE={e.mean():.3f}  epi={ep.mean():.4f}', flush=True)
        rows.append({'dataset': name, 'variant': v, 'params': acc[v]['params'],
                     'rmse_mean': r.mean(), 'rmse_std': r.std(), 'll_mean': l.mean(),
                     'ece_mean': e.mean(), 'epistemic_mean': ep.mean()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--splits', type=int, default=5, help='cap number of splits')
    ap.add_argument('--n_mc', type=int, default=100)
    ap.add_argument('--out', type=str, default='results/which_params_uci.csv')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    f = open(args.out, 'w')
    f.write('dataset,variant,params,rmse_mean,rmse_std,ll_mean,ece_mean,epistemic_mean\n')
    for name in sorted(args.datasets, key=lambda n: len(load_uci(n, 0)['x_train'])):
        for row in run_dataset(name, VARIANTS, args.splits, args.n_mc, device):
            f.write(f"{row['dataset']},{row['variant']},{row['params']},{row['rmse_mean']},"
                    f"{row['rmse_std']},{row['ll_mean']},{row['ece_mean']},{row['epistemic_mean']}\n")
        f.flush()
    f.close()
    print(f'saved {args.out}', flush=True)


if __name__ == '__main__':
    main()

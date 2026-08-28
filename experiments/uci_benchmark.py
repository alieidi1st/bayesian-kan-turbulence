"""
UCI regression benchmark: BKAN vs BNN vs KAN vs MLP and standard UQ baselines.

Follows the Hernandez-Lobato & Adams (2015) protocol (canonical datasets and
splits, z-score standardisation, RMSE and test log-likelihood in original
units). Models are compared at matched parameter budget: the BNN with one hidden
layer of HL_HIDDEN_UNITS is the reference, and the KAN/BKAN width is chosen as
the largest whose trainable-parameter count does not exceed the BNN's, so any
KAN-family advantage cannot be attributed to extra capacity.

Usage
-----
python experiments/uci_benchmark.py --datasets yacht energy --splits 5 --epochs 200
python experiments/uci_benchmark.py            # all datasets, all splits, default epochs
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bkan.data.uci import UCI_DATASETS, load_uci, n_splits, HL_HIDDEN_UNITS
from bkan.models import BNN, BayesianKAN, DeterministicKAN, DeterministicMLP, DeepEnsemble, MCDropoutNN
from bkan.training.trainer import BNNTrainer, MCDropoutTrainer, TrainingConfig


# ── Parameter counting / matching ─────────────────────────────────────────────

def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def matched_kan_width(input_dim: int, target_params: int, num=5, k=3) -> int:
    """Largest single-hidden-layer KAN width with trainable params <= target."""
    best = 1
    for width in range(1, 200):
        m = BayesianKAN(input_dim=input_dim, hidden_dims=[width], output_dim=1,
                        num=num, k=k, variant='coef', prior_type='gaussian',
                        noise='homoscedastic')
        if n_params(m) <= target_params:
            best = width
        else:
            break
    return best


# ── Metrics (Hernandez-Lobato: original units) ────────────────────────────────

_CALIB_LEVELS = np.linspace(0.1, 0.9, 9)
# z multipliers for central credible intervals of probability p: z = sqrt(2)*erfinv(p)
_CALIB_Z = (torch.erfinv(torch.tensor(_CALIB_LEVELS)) * math.sqrt(2)).numpy()


def regression_calibration_error(mean, std, y):
    """
    Expected calibration error for regression (Kuleshov et al. 2018): the mean
    absolute gap between nominal central-interval probability p and the empirical
    fraction of targets falling inside the predicted p-interval mu +/- z(p)*sigma.
    Scale-invariant, so it is computed directly in standardised units.
    """
    errs = []
    for p, z in zip(_CALIB_LEVELS, _CALIB_Z):
        covered = np.mean(np.abs(y - mean) <= z * std)
        errs.append(abs(covered - p))
    return float(np.mean(errs))


@torch.no_grad()
def evaluate(model, x_test, y_test, y_std, n_mc=100):
    """Return (RMSE, test LL, calibration error) in original target units."""
    mean, std, _ = model.predict(x_test, n_samples=n_mc)
    mean = mean.cpu().numpy().flatten()
    std = np.clip(std.cpu().numpy().flatten(), 1e-6, None)
    y = y_test.cpu().numpy().flatten()

    rmse = float(np.sqrt(np.mean((mean - y) ** 2)) * y_std)
    # Gaussian LL in normalised space, then shift to original units (- log y_std)
    ll_norm = np.mean(-0.5 * math.log(2 * math.pi) - np.log(std)
                      - 0.5 * (y - mean) ** 2 / std ** 2)
    ll = float(ll_norm - math.log(y_std))
    ece = regression_calibration_error(mean, std, y)
    return rmse, ll, ece


# ── Training dispatch ─────────────────────────────────────────────────────────

def epochs_for(n_train: int) -> int:
    """Per-dataset epoch budget: small datasets need many epochs to converge;
    large datasets need few (each epoch has many more gradient steps)."""
    if n_train <= 1000:   # yacht, boston, energy, concrete
        return 2000
    if n_train <= 2000:   # wine
        return 1500
    if n_train <= 10000:  # kin8nm, power-plant
        return 400
    if n_train <= 20000:  # naval
        return 200
    return 100            # protein


def train_model(model, x_tr, y_tr, kind, epochs, device):
    cfg = TrainingConfig(epochs=epochs, batch_size=128, learning_rate=1e-2,
                         kl_annealing_epochs=max(epochs // 4, 1), n_mc_samples=1,
                         early_stopping_patience=epochs, scheduler='cosine',
                         verbose=False)
    if kind == 'elbo':
        BNNTrainer(model, cfg, device).train(x_tr, y_tr)
    elif kind == 'ensemble':
        for member in model.members:
            MCDropoutTrainer(member, cfg, device).train(x_tr, y_tr)
    else:  # 'loss'
        MCDropoutTrainer(model, cfg, device).train(x_tr, y_tr)


# ── Model registry (matched budget) ───────────────────────────────────────────

def build_models(input_dim, hl_units, kan_width):
    """Factory dict: name -> (build_fn, train_kind)."""
    return {
        'MLP':        (lambda: DeterministicMLP(input_dim, [hl_units], 1, learn_noise=True, activation='relu'), 'loss'),
        'KAN':        (lambda: DeterministicKAN(input_dim, [kan_width], 1, noise='homoscedastic'), 'loss'),
        'BNN':        (lambda: BNN(input_dim, [hl_units], 1, prior_type='gaussian', noise='homoscedastic', activation='relu'), 'elbo'),
        'BKAN':       (lambda: BayesianKAN(input_dim, [kan_width], 1, variant='coef', prior_type='gaussian', noise='homoscedastic'), 'elbo'),
        'MCDropout':  (lambda: MCDropoutNN(input_dim, [hl_units], 1, dropout_rate=0.1, learn_noise=True, activation='relu'), 'loss'),
        'DeepEns':    (lambda: DeepEnsemble(5, input_dim, [hl_units], 1, learn_noise=True, activation='relu'), 'ensemble'),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=UCI_DATASETS)
    ap.add_argument('--splits', type=int, default=None, help='cap number of splits')
    ap.add_argument('--epochs', type=int, default=None,
                    help='fixed epochs; if omitted, use the per-dataset schedule')
    ap.add_argument('--n_mc', type=int, default=100)
    ap.add_argument('--models', nargs='+', default=None,
                    help='subset of model names to run (default: all)')
    ap.add_argument('--gap', action='store_true',
                    help='use Foong-style in-between (gap) test splits')
    ap.add_argument('--out', type=str, default='results/uci_results.csv')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Order datasets small -> large so useful results accumulate first.
    order = sorted(args.datasets, key=lambda n: len(load_uci(n, 0)['x_train']))
    print(f'device={device}  datasets={order}\n', flush=True)

    # Incremental CSV: write header once, flush after each dataset.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fcsv = open(args.out, 'w')
    fcsv.write('dataset,model,width,params,epochs,rmse_mean,rmse_std,'
               'll_mean,ll_std,ece_mean,ece_std\n')
    fcsv.flush()

    for name in order:
        ns = n_splits(name)
        if args.splits is not None:
            ns = min(ns, args.splits)
        hl = HL_HIDDEN_UNITS[name]
        d0 = load_uci(name, 0)
        ref = BNN(d0['input_dim'], [hl], 1, prior_type='gaussian', noise='homoscedastic')
        ref_params = n_params(ref)
        kan_width = matched_kan_width(d0['input_dim'], ref_params)
        epochs = args.epochs if args.epochs else epochs_for(len(d0['x_train']))

        models = build_models(d0['input_dim'], hl, kan_width)
        if args.models is not None:
            models = {m: v for m, v in models.items() if m in args.models}
        params = {m: n_params(build()) for m, (build, _) in models.items()}
        acc = {m: {'rmse': [], 'll': [], 'ece': []} for m in models}

        t0 = time.time()
        for split in range(ns):
            d = load_uci(name, split, gap=args.gap)
            x_tr = torch.tensor(d['x_train']); y_tr = torch.tensor(d['y_train'])
            x_te = torch.tensor(d['x_test']);  y_te = torch.tensor(d['y_test'])
            for mname, (build, kind) in models.items():
                torch.manual_seed(split)
                model = build().to(device)
                train_model(model, x_tr, y_tr, kind, epochs, device)
                rmse, ll, ece = evaluate(model, x_te.to(device), y_te, d['y_std'], args.n_mc)
                acc[mname]['rmse'].append(rmse)
                acc[mname]['ll'].append(ll)
                acc[mname]['ece'].append(ece)

        print(f'=== {name}  (dim={d0["input_dim"]}, N={len(d0["x_train"])}, splits={ns}, '
              f'epochs={epochs}, gap={args.gap}, BNN[{hl}]={ref_params}p, KAN width={kan_width}) ===',
              flush=True)
        print(f'{"model":12} {"params":>7} {"RMSE":>16} {"test LL":>16} {"ECE":>8}', flush=True)
        for mname in models:
            r = np.array(acc[mname]['rmse']); l = np.array(acc[mname]['ll']); e = np.array(acc[mname]['ece'])
            print(f'{mname:12} {params[mname]:>7} {r.mean():7.3f}+/-{r.std():6.3f} '
                  f'{l.mean():7.3f}+/-{l.std():6.3f} {e.mean():7.3f}', flush=True)
            fcsv.write(f'{name},{mname},{kan_width if "KAN" in mname else hl},'
                       f'{params[mname]},{epochs},{r.mean()},{r.std()},{l.mean()},{l.std()},'
                       f'{e.mean()},{e.std()}\n')
        fcsv.flush()
        print(f'    ({time.time()-t0:.0f}s)\n', flush=True)

    fcsv.close()
    print(f'saved {args.out}', flush=True)


if __name__ == '__main__':
    main()

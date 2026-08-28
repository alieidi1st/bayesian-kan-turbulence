"""
A-priori turbulence anisotropy-mapping experiment for the tensor-basis BKAN.

Trains a Bayesian tensor-basis KAN (and a deterministic point-estimate baseline)
to predict the Reynolds-stress anisotropy b_ij from RANS invariants on the
McConkey et al. (2021) curated dataset. Splits are defined in
bkan.data.turbulence.SPLITS and carry their own invariant/basis sizes:

  * cross_geometry / periodic_hills / flat_plate : 2-D flows, 2 invariants,
    3-term basis (exact for planar anisotropy);
  * square_duct : 3-D flow, 5 invariants, full 10-term Pope basis.

This script trains and evaluates only. It writes per-case predictions to
reports/predictions/*.npz and the metrics/training history to a JSON file, so
figures can be drawn from those outputs without retraining.

Usage
-----
    python experiments/anisotropy_apriori.py --split all
    python experiments/anisotropy_apriori.py --split square_duct --kl_scale 10
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bkan.data.turbulence import load_split, SPLITS
from bkan.data import tensor_basis as tb
from bkan.models.tbkan import TensorBasisBKAN, _to6, realizability_penalty

COMP = ('b11', 'b12', 'b13', 'b22', 'b23', 'b33')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDDIR = os.path.join(REPO, 'reports', 'predictions')


# ── data preparation ──────────────────────────────────────────────────────────
def prep(d, stats=None, n_inv=2, n_basis=3, grid_clip=4.0, extra=True):
    """
    Build network-ready tensors: standardised (signed-log, z-score, grid-clipped)
    input features, unit-normalised basis tensors, and the anisotropy target.

    The input features are the n_inv Pope invariants, optionally augmented with
    the extra invariant scalar features (Q, TI, PG; see tensor_basis) so the
    mapping can distinguish flow states across geometries.
    """
    lam = tb.compute_invariants(d['S'], d['R'], d['k'], d['eps'], n_inv=n_inv)
    if extra:
        q = tb.extra_scalar_features(d['S'], d['R'], d['k'], d['U'], d['gradp'])
        lam = np.concatenate([lam, q], axis=1)
    z, stats = tb.standardize(lam, stats)
    z = np.clip(z, -grid_clip, grid_clip)
    T = tb.normalize_basis(tb.compute_basis(d['S'], d['R'], d['k'], d['eps'],
                                            n_basis=n_basis))
    return (torch.tensor(z, dtype=torch.float32),
            torch.tensor(T, dtype=torch.float32),
            torch.tensor(d['b'], dtype=torch.float32), stats)


def n_features(n_inv, extra=True):
    return n_inv + (tb.N_EXTRA if extra else 0)


# ── training ──────────────────────────────────────────────────────────────────
def train(model, X, T, b, epochs, batch, lr, anneal, bayesian,
          kl_scale=1.0, seed=0):
    """Train the model; returns (model, per-epoch history)."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = len(X)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        klw = (min(1.0, ep / max(1, anneal)) * kl_scale) if bayesian else 0.0
        agg = {'loss': 0.0, 'nll': 0.0, 'realiz': 0.0, 'kl': 0.0, 'n': 0}
        for s in range(0, N, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            if bayesian:
                loss, nll, rz, kl = model.elbo_loss(X[idx], T[idx], b[idx],
                                                    kl_weight=klw, dataset_size=N)
            else:
                g = model.coefficients(X[idx], sample=False)
                bp = torch.einsum('bn,bnij->bij', g, T[idx])
                nll = ((_to6(bp) - _to6(b[idx])) ** 2).mean()
                rz = realizability_penalty(bp)
                kl = torch.zeros(())
                loss = nll + model.realiz_weight * rz
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            agg['loss'] += float(loss); agg['nll'] += float(nll)
            agg['realiz'] += float(rz); agg['kl'] += float(kl); agg['n'] += 1
        history.append({k: agg[k] / agg['n'] for k in ('loss', 'nll', 'realiz', 'kl')})
    return model, history


# ── metrics ───────────────────────────────────────────────────────────────────
def coverage_curve(err, std, levels=np.linspace(0.05, 0.95, 19)):
    from scipy.stats import norm
    z = norm.ppf(0.5 + levels / 2)
    emp = np.array([(np.abs(err) <= zz * std).mean() for zz in z])
    return levels, emp, float(np.mean(np.abs(emp - levels)))


def evaluate(model, X, T, b, bayesian, n_samples=100):
    if bayesian:
        bmean, epi, alea = model.predict(X, T, n_samples=n_samples)
        bmean, epi, alea = bmean.numpy(), epi.numpy(), alea.numpy()
        total = np.sqrt(epi ** 2 + alea ** 2)
    else:
        with torch.no_grad():
            g = model.coefficients(X, sample=False)
            bmean = _to6(torch.einsum('bn,bnij->bij', g, T)).numpy()
        epi = np.zeros_like(bmean); alea = np.zeros_like(bmean); total = None
    btrue = _to6(b).numpy()
    err = bmean - btrue
    out = {'rmse': float(np.sqrt((err ** 2).mean())),
           'rmse_per_component': {c: float(np.sqrt((err[:, i] ** 2).mean()))
                                  for i, c in enumerate(COMP)},
           'epistemic_mean': float(epi.mean()), 'aleatoric_mean': float(alea.mean())}
    if total is not None:
        # Calibration is computed only on the MEANINGFUL components: for planar
        # flows b13, b23 are identically zero and predicted ~0 with a small noise
        # sigma, so they are trivially "covered" and would otherwise dominate the
        # averaged calibration error with an artefact. A component is meaningful
        # if its DNS root-mean-square magnitude exceeds a small threshold.
        mag = np.sqrt((btrue ** 2).mean(0))
        keep = [i for i in range(6) if mag[i] > 0.02]
        e = err[:, keep].ravel(); t = total[:, keep].ravel() + 1e-12
        lev, emp, cal = coverage_curve(e, t)
        out.update({
            'calibration_error': cal,
            'meaningful_components': [COMP[i] for i in keep],
            'nll': float(0.5 * np.mean(np.log(2 * np.pi * t ** 2) + e ** 2 / t ** 2)),
            '_levels': lev.tolist(), '_coverage': emp.tolist(),
            'coverage_1sigma': [float((np.abs(err[:, i]) <= total[:, i] + 1e-12).mean())
                                for i in range(6)],
            'coverage_2sigma': [float((np.abs(err[:, i]) <= 2 * total[:, i] + 1e-12).mean())
                                for i in range(6)]})
    return out, bmean, epi, alea


# ── per-split experiment ──────────────────────────────────────────────────────
def run_split(name, args):
    cfg = SPLITS[name]
    n_inv, n_basis = cfg['n_inv'], cfg['n_basis']
    print(f'\n=== {name}  ({cfg["label"]}; n_inv={n_inv}, n_basis={n_basis}) ===',
          flush=True)
    train_d, test_d, _ = load_split(name)
    # Cap the number of test cells PER CASE: the converging-diverging grids carry
    # ~10^5 near-redundant cells, which dominate prediction cost without adding
    # statistical or visual content. A few x10^4 cells per case is ample for the
    # metrics and the field plots.
    if args.max_per_case > 0:
        rng = np.random.default_rng(0)
        keep = np.zeros(len(test_d['case']), dtype=bool)
        for c in np.unique(test_d['case']):
            idx = np.where(test_d['case'] == c)[0]
            if len(idx) > args.max_per_case:
                idx = rng.choice(idx, args.max_per_case, replace=False)
            keep[idx] = True
        test_d = {k: (v[keep] if hasattr(v, '__len__') and len(v) == len(keep) else v)
                  for k, v in test_d.items()}
    Xtr, Ttr, btr, stats = prep(train_d, n_inv=n_inv, n_basis=n_basis,
                                extra=args.extra)
    Xte, Tte, bte, _ = prep(test_d, stats, n_inv=n_inv, n_basis=n_basis,
                            extra=args.extra)
    nfeat = n_features(n_inv, args.extra)
    print(f'train {len(Xtr)}  test {len(Xte)}  features={nfeat}', flush=True)

    common = dict(n_inv=nfeat, hidden_dims=args.hidden, n_basis=n_basis,
                  realiz_weight=args.realiz)

    # Bayesian model: deterministic pre-training of the means, then VI fine-tune.
    bkan = TensorBasisBKAN(noise=args.noise, **common)
    if args.pretrain > 0:
        train(bkan, Xtr, Ttr, btr, args.pretrain, args.batch, args.lr,
              args.anneal, bayesian=False)
    _, hist = train(bkan, Xtr, Ttr, btr, args.epochs, args.batch, args.lr,
                    args.anneal, bayesian=True, kl_scale=args.kl_scale)
    m_bkan, bmean, epi, alea = evaluate(bkan, Xte, Tte, bte, bayesian=True,
                                        n_samples=args.n_samples)

    # deterministic baseline (point estimate, no posterior)
    det = TensorBasisBKAN(noise=None, **common)
    if args.pretrain > 0:
        train(det, Xtr, Ttr, btr, args.pretrain, args.batch, args.lr,
              args.anneal, bayesian=False)
    train(det, Xtr, Ttr, btr, args.epochs, args.batch, args.lr, args.anneal,
          bayesian=False)
    m_det, _, _, _ = evaluate(det, Xte, Tte, bte, bayesian=False)

    print(f'  BKAN  RMSE={m_bkan["rmse"]:.4f}  cal={m_bkan["calibration_error"]:.3f}'
          f'  epi={m_bkan["epistemic_mean"]:.4f} alea={m_bkan["aleatoric_mean"]:.4f}'
          f'  | det RMSE={m_det["rmse"]:.4f}', flush=True)

    # ── save per-case predictions for the figure/symbolic scripts ─────────────
    os.makedirs(PREDDIR, exist_ok=True)
    btrue6 = _to6(bte).numpy()
    cases = sorted(set(test_d['case']))
    for case in cases:
        sel = test_d['case'] == case
        np.savez_compressed(
            os.path.join(PREDDIR, f'{name}__{case}.npz'),
            coords=test_d['coords'][sel], b_true=btrue6[sel], b_pred=bmean[sel],
            epi=epi[sel], alea=alea[sel], dim=cfg['dim'], case=case, split=name)

    return {'split': name, 'label': cfg['label'], 'dim': cfg['dim'],
            'n_inv': n_inv, 'n_basis': n_basis, 'n_train': len(Xtr),
            'n_test': len(Xte), 'test_cases': cases, 'kl_scale': args.kl_scale,
            'history': hist, 'bkan': m_bkan, 'deterministic': m_det}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='all',
                    help="split name (see SPLITS) or 'all'")
    ap.add_argument('--hidden', type=int, nargs='+', default=[16])
    ap.add_argument('--pretrain', type=int, default=60)
    ap.add_argument('--epochs', type=int, default=250)
    ap.add_argument('--n_samples', type=int, default=50)
    ap.add_argument('--max_per_case', type=int, default=20000,
                    help='cap test cells per case (0 = use all)')
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=5e-3)
    ap.add_argument('--anneal', type=int, default=100)
    ap.add_argument('--realiz', type=float, default=10.0)
    ap.add_argument('--kl_scale', type=float, default=1.0,
                    help='KL-tempering factor (>1 widens the posterior)')
    ap.add_argument('--noise', default='heteroscedastic',
                    choices=['heteroscedastic', 'homoscedastic'])
    ap.add_argument('--extra', dest='extra', action='store_true', default=True,
                    help='augment invariants with extra scalar features (default)')
    ap.add_argument('--no_extra', dest='extra', action='store_false',
                    help='use only the Pope invariants as inputs')
    ap.add_argument('--n_basis_label', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results',
                                                  'turbulence_results.json'))
    args = ap.parse_args()

    names = list(SPLITS) if args.split == 'all' else [args.split]
    results = [run_split(n, args) for n in names]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nwrote {args.out} and predictions to {PREDDIR}', flush=True)


if __name__ == '__main__':
    main()

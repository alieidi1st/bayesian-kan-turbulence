# Bayesian Kolmogorov–Arnold networks for turbulence closures

This repository contains the reference implementation for a Bayesian
tensor-basis Kolmogorov–Arnold network (KAN) that predicts the Reynolds-stress
anisotropy for RANS turbulence closures and reports a calibrated estimate of its
own uncertainty. It accompanies the paper

> A. Eidi, K. Jigjid, T. Buchanan, R. P. Dwight,
> *Bayesian Kolmogorov–Arnold networks for trustworthy Reynolds-stress closures*.

The idea is to keep two properties that data-driven closures usually trade
against each other. A KAN places learnable univariate spline functions on the
network edges, which makes the mapping inspectable and open to symbolic
distillation. Putting variational posteriors over those splines then turns the
closure into a Bayesian model, so every prediction comes with epistemic and
aleatoric uncertainty rather than a bare point estimate. The tensor-basis
construction (Pope 1975) makes the predicted anisotropy invariant, symmetric and
traceless by construction, and a realizability penalty keeps it inside the
physically attainable region.

The code reproduces the three quantitative parts of the paper that rely on
public data:

- a controlled regression benchmark on the standard UCI datasets, where the
  Bayesian KAN is compared against a variational BNN, MC-dropout, a deep
  ensemble and deterministic baselines at matched parameter budgets;
- the *a-priori* Reynolds-stress anisotropy study on the McConkey et al. (2021)
  curated turbulence dataset, across held-out geometries and Reynolds numbers;
- the spatially correlated aleatoric-noise generator used to propagate the
  posterior through a flow solver.

The wind-farm LES and the solver-propagation cases are too large to distribute;
see [docs/DATA.md](docs/DATA.md) for what is and is not included.

## Installation

The code targets Python 3.9+ and PyTorch 2.x.

```bash
git clone https://github.com/alieidi1st/bayesian-kan-turbulence.git
cd bayesian-kan-turbulence
python -m venv .venv && source .venv/bin/activate    # or conda
pip install -r requirements.txt
pip install -e .                                      # makes `import bkan` work anywhere
```

The Bayesian KAN layer reuses the B-spline basis from
[`pykan`](https://github.com/KindXiaoming/pykan) (Liu et al. 2024), pulled in as
a dependency.

## Repository layout

```
bkan/
  models/         KAN layer, Bayesian KAN, tensor-basis BKAN, and baselines
  data/           toy problems, UCI loaders, turbulence dataset + Pope basis
  training/       variational trainers, ELBO and KL annealing
  propagation/    spatially correlated aleatoric noise for solver propagation
  viz/            plotting helpers
experiments/      command-line drivers for the paper results
notebooks/        1-D validation, toy problems, and edge interpretability
docs/DATA.md      how to obtain the datasets
```

## Data

The datasets are downloaded separately. In short: the UCI splits come from the
`DropoutUncertaintyExps` repository and go under `data_uci/`, and the turbulence
dataset comes from Kaggle (`ryleymcconkey/ml-turbulence-dataset`); only its two
CSV files are needed. Full instructions, including the expected directory
layout, are in [docs/DATA.md](docs/DATA.md).

## Reproducing the results

### 1. Regression benchmark (UCI)

```bash
# a quick pass on two datasets, then the full run
python experiments/uci_benchmark.py --datasets yacht energy --splits 5 --epochs 200
python experiments/uci_benchmark.py            # all datasets, all splits
```

This trains the Bayesian KAN and the baselines with matched parameter counts and
writes RMSE, test log-likelihood and calibration error to `results/`. The
companion ablation

```bash
python experiments/which_params_uci.py
```

varies which network parameters carry the posterior (spline coefficients only,
scale, or the full set) and reports the effect on accuracy and calibration.

### 2. A-priori anisotropy mapping

```bash
python experiments/anisotropy_apriori.py --split all
```

trains the tensor-basis Bayesian KAN and a deterministic baseline on every
train/test split (cross-geometry, periodic-hills, flat-plate, and the 3-D square
duct with the full 10-term basis), and saves per-case predictions and metrics —
RMSE, realizability violation, calibration coverage and NLL — to `results/` and
`reports/predictions/`. The turbulence which-params ablation is

```bash
python experiments/which_params_anisotropy.py
```

### 3. Symbolic distillation

```bash
python experiments/symbolic_recovery.py --split periodic_hills
```

fits a trained edge function with a small library of candidate expressions to
show how the learned closure can be read as a compact formula.

### Notebooks

The notebooks in `notebooks/` are self-contained and need no external data. They
cover the 1-D uncertainty decomposition, the toy-problem comparison against a
BNN, and the edge-interpretability figures.

## The uncertainty-propagation module

`bkan.propagation` turns a per-point aleatoric standard deviation into spatially
correlated noise samples. Independent per-cell sampling would look like white
noise and largely average out inside an elliptic solve, so the field is
correlated over a physical length scale instead. Connectivity is built from the
cell-centre coordinates with a KD-tree, so the module works on any mesh or point
cloud without needing the mesh topology.

```python
import numpy as np
from bkan.propagation import (
    build_adjacency_from_coordinates,
    generate_correlated_aleatoric_sample,
)

cell_centers = np.random.rand(2000, 3)          # (n_cells, 3)
sigma = 0.05 * np.ones(len(cell_centers))        # aleatoric std from the model
mask = np.ones(len(cell_centers), dtype=bool)    # where the correction is active

adjacency = build_adjacency_from_coordinates(cell_centers, k_neighbors=10)
sample = generate_correlated_aleatoric_sample(
    sigma, mask, adjacency, smooth_iterations=20, random_state=0
)
```

Drawing many such samples, injecting them alongside the posterior parameter
draws, and running the solver once per sample gives the propagated uncertainty
band reported in the paper.

## Citation

If you use this code, please cite the paper (see [CITATION.cff](CITATION.cff)):

```bibtex
@article{eidi2026bkan,
  title   = {Bayesian Kolmogorov--Arnold networks for trustworthy Reynolds-stress closures},
  author  = {Eidi, Ali and Jigjid, Kherlen and Buchanan, Tyler and Dwight, Richard P.},
  year    = {2026}
}
```

## License

Released under the MIT License; see [LICENSE](LICENSE). The bundled dependency
`pykan` is also MIT-licensed.

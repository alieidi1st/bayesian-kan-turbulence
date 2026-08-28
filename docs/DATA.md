# Datasets

None of the datasets are bundled with the repository. The two public ones are
downloaded from their original sources with the instructions below; the
wind-farm LES is not redistributed here (see the last section).

## 1. UCI regression benchmark

The controlled-regression experiments use the nine UCI datasets with the exact
train/test splits from Hernández-Lobato & Adams (2015), as packaged in Yarin
Gal's `DropoutUncertaintyExps` repository. Using the same splits keeps our
numbers directly comparable to the published baselines.

```bash
git clone https://github.com/yaringal/DropoutUncertaintyExps.git
mkdir -p data_uci
cp -r DropoutUncertaintyExps/UCI_Datasets/* data_uci/
```

After this you should have, for each dataset, a directory such as
`data_uci/yacht/data/` containing `data.txt`, `index_features.txt`,
`index_target.txt`, `index_train_<k>.txt`, `index_test_<k>.txt` and
`n_splits.txt`. The nine directories expected are:

```
bostonHousing  concrete  energy  kin8nm  naval-propulsion-plant
power-plant    protein-tertiary-structure  wine-quality-red  yacht
```

`bkan.data.uci.load_uci` reads from `data_uci/` at the repository root by
default; pass `root=` to point elsewhere.

## 2. Turbulence-anisotropy dataset

The Reynolds-stress anisotropy study uses the curated RANS/high-fidelity dataset
of McConkey, Yee & Lien (2021), *A curated dataset for data-driven turbulence
modelling* (Scientific Data), hosted on Kaggle as
`ryleymcconkey/ml-turbulence-dataset`.

Only two CSV files are needed for the *a-priori* study — the RANS features and
the high-fidelity labels — roughly 850 MB together. The full `foam/` OpenFOAM
cases (~69 GB) are **not** required.

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('ryleymcconkey/ml-turbulence-dataset'))"
```

`bkan.data.turbulence.load_split` looks in the kagglehub cache
(`~/.cache/kagglehub/datasets/ryleymcconkey/ml-turbulence-dataset/versions/6`)
by default; pass `root=` to point at wherever you placed `komegasst.csv` and
`REF.csv`. The named train/test splits are `cross_geometry`, `periodic_hills`,
`flat_plate` and `square_duct` (the last is the 3-D, 10-term-basis case).

## 3. Wind-farm LES and solver propagation

The a-posteriori study propagates the closure through a RANS solver on
three-dimensional wind-farm cases and compares against LES. Those meshes,
LES references and per-sample correction fields are tens of gigabytes and are
not distributed here.

What *is* included is the machinery that makes the propagation reproducible: the
spatially correlated aleatoric-noise generator in `bkan.propagation`, which
turns a per-point noise standard deviation into correlated samples on any point
cloud. It is mesh-agnostic (it builds connectivity from cell-centre coordinates
with a KD-tree), so it can be applied to any solver's cell centres. A minimal,
data-free demonstration on a synthetic point cloud is given in the README.

"""Bayesian tensor-basis Kolmogorov--Arnold networks with quantified uncertainty.

The package is organised as:

- ``bkan.models``       -- KAN layers, the Bayesian KAN and tensor-basis KAN, and
                           the baseline models used for comparison.
- ``bkan.data``         -- toy regression problems, the UCI loaders, and the
                           turbulence-anisotropy dataset and Pope tensor basis.
- ``bkan.training``     -- variational trainers and the ELBO/KL schedule.
- ``bkan.propagation``  -- spatially correlated aleatoric noise for uncertainty
                           propagation through a flow solver.
- ``bkan.viz``          -- plotting helpers used by the experiments.
"""

__version__ = "0.1.0"

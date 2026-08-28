"""
Symbolic regression utilities for Bayesian KAN interpretability.

Follows the workflow of Liu et al. (2024) "KAN: Kolmogorov-Arnold Networks"
(ICLR 2025) and the reference implementation in pykan: for each learned
activation function, fit a library of candidate symbolic functions using the
four-parameter affine model

    y ~= c * f(a * x + b) + d

and rank candidates by their coefficient of determination R^2.  The input
transform (a, b) is found by an iterative grid search (matching pykan's
`fit_params`); the output transform (c, d) is solved by linear least squares.

The Bayesian extension (this work) adds pointwise uncertainty: each activation
is a distribution phi(x) ~ N(mu(x), sigma(x)^2).  We fit the symbolic form to
the posterior mean mu(x) and additionally report how well that same fitted form
explains the +/- 2 sigma posterior bounds:

  - r2_mean  : R^2 of the fit against the posterior mean mu(x)
  - r2_lower : R^2 of the fitted form against mu(x) - 2 sigma(x)
  - r2_upper : R^2 of the fitted form against mu(x) + 2 sigma(x)

A high r2_mean with a small spread (r2_upper close to r2_lower) indicates a
confident symbolic identification; a large spread indicates the symbolic label
is sensitive to coefficient uncertainty.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Symbolic function library ─────────────────────────────────────────────────
# Functions take the (input-transformed) argument t = a*x + b. The 15 entries
# follow the common math/physics library used in Liu et al. (2024).

SYMBOLIC_LIB: Dict[str, callable] = {
    'x':           lambda t: t,
    'x^2':         lambda t: t ** 2,
    'x^3':         lambda t: t ** 3,
    'x^4':         lambda t: t ** 4,
    '1/x':         lambda t: 1.0 / (t + 1e-4 * np.sign(t) + 1e-8),
    'sqrt(x)':     lambda t: np.sqrt(np.abs(t)),
    '1/sqrt(x)':   lambda t: 1.0 / (np.sqrt(np.abs(t)) + 1e-4),
    'exp(x)':      lambda t: np.exp(np.clip(t, -20, 20)),
    'log(|x|)':    lambda t: np.log(np.abs(t) + 1e-4),
    'sin(x)':      lambda t: np.sin(t),
    'tanh(x)':     lambda t: np.tanh(t),
    'abs(x)':      lambda t: np.abs(t),
    'sgn(x)':      lambda t: np.sign(t),
    'arctan(x)':   lambda t: np.arctan(t),
    'x*exp(-x^2)': lambda t: t * np.exp(-np.clip(t ** 2, 0, 40)),
}

# Complexity score per function, used as a tie-breaker when several candidates
# achieve similar R^2 (cf. pykan `auto_symbolic` weight_simple). Lower is
# simpler. Over a bounded interval many functions are mutually approximable
# (e.g. a sine arc resembles a parabola), so among near-equal fits the simpler
# symbolic form is preferred.
SYMBOLIC_COMPLEXITY: Dict[str, int] = {
    'x': 1,
    'x^2': 2, 'abs(x)': 2, 'sgn(x)': 2,
    'x^3': 3, 'sqrt(x)': 3, '1/x': 3,
    'sin(x)': 4, 'exp(x)': 4, 'tanh(x)': 4, 'arctan(x)': 4,
    'x^4': 4, '1/sqrt(x)': 4, 'log(|x|)': 4,
    'x*exp(-x^2)': 5,
}


# ── Core fitting routine ──────────────────────────────────────────────────────

def _r2_of_input_transform(
    x: np.ndarray,
    y: np.ndarray,
    f: callable,
    a_grid: np.ndarray,
    b_grid: np.ndarray,
) -> np.ndarray:
    """
    R^2 of the best linear fit c*f(a*x+b)+d, evaluated over an (a, b) grid.

    The R^2 of a linear fit equals the squared Pearson correlation between
    f(a*x+b) and y, so c and d need not be solved explicitly here.

    Returns
    -------
    r2 : np.ndarray, shape (len(a_grid), len(b_grid))
    """
    # post[k, i, j] = f(a_i * x_k + b_j)
    arg = a_grid[None, :, None] * x[:, None, None] + b_grid[None, None, :]
    post = f(arg)
    post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0)

    post_mean = post.mean(axis=0, keepdims=True)
    y_mean = y.mean()
    cov = np.sum((post - post_mean) * (y - y_mean)[:, None, None], axis=0) ** 2
    var = (np.sum((post - post_mean) ** 2, axis=0)
           * np.sum((y - y_mean) ** 2))
    r2 = cov / (var + 1e-12)
    return np.nan_to_num(r2)


def _fit_affine(
    x: np.ndarray,
    y: np.ndarray,
    f: callable,
    a_range: Tuple[float, float] = (-6.0, 6.0),
    b_range: Tuple[float, float] = (-6.0, 6.0),
    grid_number: int = 31,
    iterations: int = 3,
) -> Tuple[float, float, float, float, float]:
    """
    Fit y ~= c * f(a*x + b) + d by iterative (a, b) grid search + linear (c, d).

    Mirrors pykan's `fit_params`: sweep the input transform (a, b) on a grid,
    zoom in around the best point over several iterations, then solve the
    output transform (c, d) by ordinary least squares.

    Returns
    -------
    a, b, c, d : float   fitted parameters
    r2         : float   coefficient of determination on (x, y)
    """
    a_lo, a_hi = a_range
    b_lo, b_hi = b_range
    a_id = b_id = 0
    a_vals = b_vals = None
    r2 = None

    for _ in range(iterations):
        a_vals = np.linspace(a_lo, a_hi, grid_number)
        b_vals = np.linspace(b_lo, b_hi, grid_number)
        r2 = _r2_of_input_transform(x, y, f, a_vals, b_vals)

        best = np.argmax(r2)
        a_id, b_id = np.unravel_index(best, r2.shape)

        # Zoom in around the best (a, b)
        a_lo, a_hi = a_vals[max(a_id - 1, 0)], a_vals[min(a_id + 1, grid_number - 1)]
        b_lo, b_hi = b_vals[max(b_id - 1, 0)], b_vals[min(b_id + 1, grid_number - 1)]
        if a_lo == a_hi:
            a_lo, a_hi = a_range
        if b_lo == b_hi:
            b_lo, b_hi = b_range

    a_best = float(a_vals[a_id])
    b_best = float(b_vals[b_id])

    # Solve output transform c, d by least squares: y ~= c*post + d
    post = np.nan_to_num(f(a_best * x + b_best), nan=0.0, posinf=0.0, neginf=0.0)
    A = np.column_stack([post, np.ones_like(post)])
    (c_best, d_best), *_ = np.linalg.lstsq(A, y, rcond=None)

    y_pred = c_best * post + d_best
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_best = 1.0 - ss_res / (ss_tot + 1e-12)

    return a_best, b_best, float(c_best), float(d_best), float(r2_best)


def _r2_against_fit(
    x: np.ndarray,
    y: np.ndarray,
    f: callable,
    a: float, b: float, c: float, d: float,
) -> float:
    """R^2 of a fixed fitted symbolic curve c*f(a*x+b)+d against target y."""
    post = np.nan_to_num(f(a * x + b), nan=0.0, posinf=0.0, neginf=0.0)
    y_pred = c * post + d
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


# ── Public API ────────────────────────────────────────────────────────────────

def suggest_symbolic(
    x_grid: np.ndarray,
    y_mean: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    n_top: int = 5,
    lib: Optional[Dict[str, callable]] = None,
    r2_tol: float = 0.01,
) -> List[dict]:
    """
    Rank symbolic functions by four-parameter affine fit to the activation.

    For each candidate f, fits y_mean(x) ~= c * f(a*x + b) + d (Liu et al. 2024,
    pykan `fit_params`).  When y_std is provided, the fitted form is also scored
    against the +/- 2 sigma posterior bounds.

    Candidates are ranked primarily by R^2, with a complexity tie-breaker: any
    candidate whose R^2 is within `r2_tol` of the best is promoted ahead of more
    complex functions (cf. pykan `auto_symbolic`). This favours the simpler
    symbolic form when several functions fit comparably well over the (bounded)
    activation domain.

    Parameters
    ----------
    x_grid : np.ndarray, shape (n,)
        Input values.
    y_mean : np.ndarray, shape (n,)
        Posterior mean of the activation function.
    y_std : np.ndarray, shape (n,), optional
        Pointwise posterior standard deviation (from `get_activation_stats`).
    n_top : int
        Number of top candidates to return. Default: 5.
    lib : dict, optional
        Custom function library. Defaults to SYMBOLIC_LIB.
    r2_tol : float
        R^2 tolerance for the complexity tie-breaker. Default: 0.01.

    Returns
    -------
    results : list of dict, length n_top.
        Keys: 'name', 'formula', 'r2_mean', 'r2_lower', 'r2_upper',
        'complexity', 'a', 'b', 'c', 'd'.
    """
    if lib is None:
        lib = SYMBOLIC_LIB

    rows = []
    for name, f in lib.items():
        a, b, c, d, r2m = _fit_affine(x_grid, y_mean, f)

        r2l = r2u = None
        if y_std is not None:
            r2l = _r2_against_fit(x_grid, y_mean - 2.0 * y_std, f, a, b, c, d)
            r2u = _r2_against_fit(x_grid, y_mean + 2.0 * y_std, f, a, b, c, d)

        arg = f'({a:+.3g}*x{b:+.3g})'
        # substitute only the standalone variable x (not the x inside e.g. 'exp')
        body = re.sub(r'(?<![A-Za-z])x(?![A-Za-z])', arg, name)
        formula = f'{c:+.3g} * {body} {d:+.3g}'
        rows.append({
            'name':     name,
            'formula':  formula,
            'r2_mean':  r2m,
            'r2_lower': r2l,
            'r2_upper': r2u,
            'complexity': SYMBOLIC_COMPLEXITY.get(name, 5),
            'a': a, 'b': b, 'c': c, 'd': d,
        })

    # Primary rank by R^2; complexity tie-break within r2_tol of the best.
    best_r2 = max(r['r2_mean'] for r in rows)
    rows.sort(key=lambda r: (
        -(r['r2_mean'] >= best_r2 - r2_tol),  # within-tolerance group first
        r['complexity'] if r['r2_mean'] >= best_r2 - r2_tol else 0,  # simplest in group
        -r['r2_mean'],                         # then by R^2
    ))
    return rows[:n_top]


def format_symbolic_table(results: List[dict]) -> str:
    """
    Format suggest_symbolic output as a human-readable table.

    Parameters
    ----------
    results : list of dict
        Output of suggest_symbolic.

    Returns
    -------
    table : str
    """
    has_bounds = results[0]['r2_lower'] is not None
    header = f"{'Rank':>4}  {'Function':>12}  {'R2_mean':>8}"
    if has_bounds:
        header += f"  {'R2_lower':>8}  {'R2_upper':>8}"
    header += "  Formula"
    lines = [header, '-' * len(header)]

    for rank, row in enumerate(results, 1):
        line = f"{rank:>4}  {row['name']:>12}  {row['r2_mean']:>8.4f}"
        if has_bounds:
            line += f"  {row['r2_lower']:>8.4f}  {row['r2_upper']:>8.4f}"
        line += f"  {row['formula']}"
        lines.append(line)

    return '\n'.join(lines)

"""
Spatial correlation utilities for generating correlated aleatoric noise fields.

These utilities are for stochastic propagation of uncertainty in CFD simulations.
When propagating aleatoric uncertainty, independent sampling per cell creates
white noise. These functions generate spatially correlated noise instead.

Usage:
------
    from bkan.propagation import (
        build_adjacency_from_coordinates,
        laplacian_smooth,
        generate_correlated_aleatoric_sample,
    )

    # Build mesh connectivity from cell centers
    cell_centers = np.column_stack([x, y, z])
    adjacency = build_adjacency_from_coordinates(cell_centers, k_neighbors=10)

    # Generate correlated noise for correction regions
    noise = generate_correlated_aleatoric_sample(
        sigma_aleatoric=sigma_aleatoric,
        correction_mask=correction_mask,
        mesh_adjacency=adjacency,
        smooth_iterations=30
    )
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from typing import Optional


def build_adjacency_from_coordinates(
    cell_centers: np.ndarray,
    k_neighbors: int = 10
) -> sparse.csr_matrix:
    """
    Build approximate mesh connectivity from cell center coordinates.

    Uses k-nearest neighbors to approximate the true mesh connectivity.
    This is useful when extracting actual connectivity from OpenFOAM
    is not convenient.

    Args:
        cell_centers: (n_cells, 3) array of x, y, z positions
        k_neighbors: Number of nearest neighbors to connect.
                     Typical values:
                     - Hexahedral mesh: 6
                     - Tetrahedral mesh: 10-15
                     - Polyhedral mesh: 10-20
                     Use slightly more than actual for safety.

    Returns:
        Sparse CSR adjacency matrix (n_cells, n_cells)
    """
    n_cells = len(cell_centers)

    # Build k-d tree for fast neighbor search
    tree = cKDTree(cell_centers)

    # Find k nearest neighbors for each cell
    # First neighbor is the cell itself (distance=0), so query k+1
    distances, indices = tree.query(cell_centers, k=k_neighbors + 1)

    # Build sparse matrix (skip self-connections)
    rows = np.repeat(np.arange(n_cells), k_neighbors)
    cols = indices[:, 1:].flatten()  # Skip column 0 (self)

    adjacency = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_cells, n_cells)
    )

    return adjacency


def build_adjacency_distance_cutoff(
    cell_centers: np.ndarray,
    cutoff_distance: float
) -> sparse.csr_matrix:
    """
    Build mesh connectivity by connecting all cells within a cutoff distance.

    Alternative to k-nearest neighbors when you want distance-based connectivity.

    Args:
        cell_centers: (n_cells, 3) array of x, y, z positions
        cutoff_distance: Maximum distance to consider cells as neighbors.
                        Should be ~1.5x typical cell size.

    Returns:
        Sparse CSR adjacency matrix (n_cells, n_cells)
    """
    tree = cKDTree(cell_centers)

    # Find all pairs within cutoff
    pairs = tree.query_pairs(cutoff_distance, output_type='ndarray')

    if len(pairs) == 0:
        n_cells = len(cell_centers)
        return sparse.csr_matrix((n_cells, n_cells))

    # Make symmetric (both directions)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])

    n_cells = len(cell_centers)
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n_cells, n_cells)
    )

    return adjacency


def laplacian_smooth(
    values: np.ndarray,
    adjacency: sparse.csr_matrix,
    iterations: int = 30,
    diffusion: float = 0.5
) -> np.ndarray:
    """
    Smooth values on unstructured mesh using iterative Laplacian smoothing.

    This creates spatial correlation by averaging with neighbors iteratively.
    More iterations = longer correlation length = smoother result.

    Args:
        values: (n_cells,) array of values to smooth
        adjacency: Sparse adjacency matrix from build_adjacency_*
        iterations: Number of smoothing iterations.
                   More iterations = smoother/longer correlation.
                   Typical: 20-50
        diffusion: Mixing factor between 0 and 1.
                  0 = no smoothing, 1 = full neighbor average.
                  Typical: 0.5

    Returns:
        Smoothed values with spatial correlation
    """
    values = values.copy()

    # Compute degree (number of neighbors per cell)
    degree = np.array(adjacency.sum(axis=1)).flatten()
    degree[degree == 0] = 1  # Avoid division by zero for isolated cells

    for _ in range(iterations):
        # Compute neighbor average
        neighbor_sum = adjacency @ values
        neighbor_avg = neighbor_sum / degree

        # Mix original with neighbor average
        values = (1 - diffusion) * values + diffusion * neighbor_avg

    return values


def generate_correlated_aleatoric_sample(
    sigma_aleatoric: np.ndarray,
    correction_mask: np.ndarray,
    mesh_adjacency: sparse.csr_matrix,
    smooth_iterations: int = 30,
    diffusion: float = 0.5,
    taper_boundaries: bool = True,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate spatially correlated aleatoric noise for correction regions.

    This function:
    1. Extracts the correction region submesh
    2. Generates white noise scaled by aleatoric uncertainty
    3. Smooths it to create spatial correlation
    4. Optionally tapers at boundaries to avoid discontinuities

    Args:
        sigma_aleatoric: (n_cells,) aleatoric std from BNN prediction
        correction_mask: (n_cells,) boolean array, True = correction region
        mesh_adjacency: Sparse adjacency matrix for full mesh
        smooth_iterations: Number of Laplacian smoothing iterations
        diffusion: Smoothing diffusion parameter (0-1)
        taper_boundaries: If True, reduce noise at correction region boundaries
        random_state: Random seed for reproducibility

    Returns:
        (n_cells,) correlated noise field (zero outside correction regions)

    Example:
        >>> # Build adjacency from cell centers
        >>> adjacency = build_adjacency_from_coordinates(cell_centers)
        >>>
        >>> # Generate 100 correlated samples for stochastic propagation
        >>> samples = []
        >>> for i in range(100):
        ...     epistemic_sample = bnn_mc_samples[i]  # From BNN weight sampling
        ...     aleatoric_noise = generate_correlated_aleatoric_sample(
        ...         sigma_aleatoric, correction_mask, adjacency, random_state=i
        ...     )
        ...     total_sample = epistemic_sample + aleatoric_noise
        ...     samples.append(total_sample)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_cells = len(sigma_aleatoric)

    # Extract correction cell indices
    corr_idx = np.where(correction_mask)[0]
    n_corr = len(corr_idx)

    if n_corr == 0:
        return np.zeros(n_cells)

    # Build mapping from global to local indices
    global_to_local = {g: l for l, g in enumerate(corr_idx)}

    # Extract submesh adjacency for correction cells only
    rows, cols = [], []
    boundary_cells = set()

    for local_i, global_i in enumerate(corr_idx):
        # Get neighbors from full mesh adjacency
        start, end = mesh_adjacency.indptr[global_i], mesh_adjacency.indptr[global_i + 1]
        neighbors = mesh_adjacency.indices[start:end]

        has_non_correction_neighbor = False
        for global_j in neighbors:
            if global_j in global_to_local:
                local_j = global_to_local[global_j]
                rows.append(local_i)
                cols.append(local_j)
            else:
                has_non_correction_neighbor = True

        if has_non_correction_neighbor:
            boundary_cells.add(local_i)

    # Build sparse submesh adjacency
    if len(rows) > 0:
        sub_adjacency = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_corr, n_corr)
        )
    else:
        # No connectivity - return unsmoothed noise
        result = np.zeros(n_cells)
        result[corr_idx] = np.random.randn(n_corr) * sigma_aleatoric[corr_idx]
        return result

    # Generate white noise scaled by aleatoric std
    white_noise = np.random.randn(n_corr) * sigma_aleatoric[corr_idx]

    # Smooth to create spatial correlation
    smoothed = laplacian_smooth(white_noise, sub_adjacency, smooth_iterations, diffusion)

    # Optionally taper at boundaries
    if taper_boundaries and len(boundary_cells) > 0:
        taper = np.ones(n_corr)
        for local_i in boundary_cells:
            taper[local_i] = 0.5
        smoothed *= taper

    # Map back to full domain
    result = np.zeros(n_cells)
    result[corr_idx] = smoothed

    return result


def random_fourier_features_sample(
    cell_centers: np.ndarray,
    sigma_aleatoric: np.ndarray,
    length_scale: float,
    n_features: int = 100,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate correlated noise using Random Fourier Features approximation.

    This approximates a Gaussian Process with squared exponential kernel
    using random Fourier features. More memory efficient than full GP
    for large meshes.

    Args:
        cell_centers: (n_cells, 3) array of x, y, z positions
        sigma_aleatoric: (n_cells,) aleatoric std from BNN
        length_scale: Correlation length scale (physical units)
        n_features: Number of random features (more = better approximation)
        random_state: Random seed for reproducibility

    Returns:
        (n_cells,) correlated noise field
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_cells = len(cell_centers)
    n_dims = cell_centers.shape[1]

    # Random frequencies
    W = np.random.randn(n_dims, n_features) / length_scale
    b = np.random.uniform(0, 2 * np.pi, n_features)

    # Compute features: Z = sqrt(2/D) * cos(X @ W + b)
    Z = np.sqrt(2.0 / n_features) * np.cos(cell_centers @ W + b)

    # Sample in feature space
    theta = np.random.randn(n_features)

    # Project to cell space and scale by aleatoric
    correlated_noise = (Z @ theta) * sigma_aleatoric

    return correlated_noise

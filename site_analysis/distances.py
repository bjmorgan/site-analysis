"""Minimum-image distance functions for periodic systems.

Provides distance calculations operating on numpy arrays and a lattice
matrix, with no pymatgen dependency. Optional numba acceleration for
single-pair distances.
"""

from __future__ import annotations

import numpy as np

from site_analysis._compat import HAS_NUMBA


# 27 shift vectors for periodic image search: {-1, 0, 1}^3
_SHIFTS_27 = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.float64,
)


if HAS_NUMBA:
    import numba  # type: ignore

    @numba.njit(cache=True)  # type: ignore[misc]
    def _mic_distance_numba(
        frac1: np.ndarray,
        frac2: np.ndarray,
        lattice_matrix: np.ndarray,
    ) -> float:
        """JIT-compiled minimum-image distance checking all 27 images.

        Args:
            frac1: Fractional coordinates of point 1, shape (3,).
            frac2: Fractional coordinates of point 2, shape (3,).
            lattice_matrix: (3, 3) lattice matrix.

        Returns:
            The minimum-image distance.
        """
        d_frac_base = np.empty(3)
        for i in range(3):
            d_frac_base[i] = frac1[i] - frac2[i]

        min_dist_sq = np.inf
        for si in range(-1, 2):
            for sj in range(-1, 2):
                for sk in range(-1, 2):
                    d0 = d_frac_base[0] + si
                    d1 = d_frac_base[1] + sj
                    d2 = d_frac_base[2] + sk
                    # Convert to Cartesian: d_frac @ lattice_matrix
                    cx = d0 * lattice_matrix[0, 0] + d1 * lattice_matrix[1, 0] + d2 * lattice_matrix[2, 0]
                    cy = d0 * lattice_matrix[0, 1] + d1 * lattice_matrix[1, 1] + d2 * lattice_matrix[2, 1]
                    cz = d0 * lattice_matrix[0, 2] + d1 * lattice_matrix[1, 2] + d2 * lattice_matrix[2, 2]
                    dist_sq = cx * cx + cy * cy + cz * cz
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
        return min_dist_sq ** 0.5


def mic_distance(
    frac1: np.ndarray,
    frac2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> float:
    """Minimum-image distance between two points in a periodic cell.

    Checks all 27 periodic images to find the true minimum distance.
    Uses numba JIT compilation when available for improved performance
    on repeated single-pair calls.

    Args:
        frac1: Fractional coordinates of point 1, shape (3,).
        frac2: Fractional coordinates of point 2, shape (3,).
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention: ``lattice.matrix``).

    Returns:
        The minimum-image distance in Angstroms.
    """
    if HAS_NUMBA:
        return _mic_distance_numba(frac1, frac2, lattice_matrix)
    d_frac = frac1 - frac2
    # (27, 3) shifted difference vectors
    d_frac_all = d_frac + _SHIFTS_27
    # Convert to Cartesian and compute norms
    d_cart_all = d_frac_all @ lattice_matrix
    return float(np.min(np.linalg.norm(d_cart_all, axis=1)))


def all_mic_distances(
    frac_coords1: np.ndarray,
    frac_coords2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Minimum-image distance matrix between two sets of points.

    Checks all 27 periodic images per pair to find true minimum
    distances, which is necessary for triclinic cells.

    Args:
        frac_coords1: Fractional coordinates, shape (N, 3).
        frac_coords2: Fractional coordinates, shape (M, 3).
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention: ``lattice.matrix``).

    Returns:
        (N, M) array of minimum-image distances in Angstroms.
    """
    if frac_coords1.shape[0] == 0 or frac_coords2.shape[0] == 0:
        return np.empty((frac_coords1.shape[0], frac_coords2.shape[0]))
    # (N, 1, 3) - (1, M, 3) -> (N, M, 3) difference vectors
    d_frac = frac_coords1[:, np.newaxis, :] - frac_coords2[np.newaxis, :, :]
    # Loop over 27 shifts with running minimum to avoid (N, M, 27, 3) allocation
    min_dist_sq = np.full(
        (frac_coords1.shape[0], frac_coords2.shape[0]), np.inf
    )
    for shift in _SHIFTS_27:
        d_cart = (d_frac + shift) @ lattice_matrix  # (N, M, 3)
        dist_sq = np.einsum("ijk,ijk->ij", d_cart, d_cart)  # (N, M)
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)
    return np.sqrt(min_dist_sq)


def frac_to_cart(
    frac_coords: np.ndarray,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Convert fractional coordinates to Cartesian coordinates.

    Args:
        frac_coords: Fractional coordinates, shape (3,) or (N, 3).
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention: ``lattice.matrix``).

    Returns:
        Cartesian coordinates with the same shape as the input.
    """
    return frac_coords @ lattice_matrix

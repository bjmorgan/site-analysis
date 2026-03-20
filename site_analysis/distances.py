"""Minimum-image distance functions for periodic systems.

Provides distance calculations operating on numpy arrays and a lattice
matrix, with no pymatgen dependency. Optional numba acceleration for
single-pair distances.
"""

from __future__ import annotations

import numpy as np


# 27 shift vectors for periodic image search: {-1, 0, 1}^3
_SHIFTS_27 = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.float64,
)


def mic_distance(
    frac1: np.ndarray,
    frac2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> float:
    """Minimum-image distance between two points in a periodic cell.

    Checks all 27 periodic images to find the true minimum distance,
    which is necessary for triclinic cells where the simple ``round``
    approach can select the wrong image.

    Args:
        frac1: Fractional coordinates of point 1, shape (3,).
        frac2: Fractional coordinates of point 2, shape (3,).
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention: ``lattice.matrix``).

    Returns:
        The minimum-image distance in Angstroms.
    """
    d_frac = frac1 - frac2
    # (27, 3) shifted difference vectors
    d_frac_all = d_frac + _SHIFTS_27
    # Convert to Cartesian and compute norms
    d_cart_all = d_frac_all @ lattice_matrix
    return float(np.min(np.linalg.norm(d_cart_all, axis=1)))


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

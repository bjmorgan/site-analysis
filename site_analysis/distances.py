"""Minimum-image distance functions for periodic systems.

Provides distance calculations operating on numpy arrays and a lattice
matrix, with no pymatgen dependency. Optional numba acceleration for
single-pair distances.
"""

from __future__ import annotations

import numpy as np


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

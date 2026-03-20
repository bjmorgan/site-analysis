"""Minimum-image distance and coordinate conversion functions for periodic systems.

Provides distance calculations and fractional-to-Cartesian coordinate
conversion operating on numpy arrays and a lattice matrix, with no
pymatgen dependency. Optional numba acceleration for single-pair and
batch distances.
"""

from __future__ import annotations

import numpy as np

from site_analysis._compat import HAS_NUMBA


# 27 shift vectors for periodic image search: {-1, 0, 1}^3
_SHIFTS_27 = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.float64,
)
_SHIFTS_27.flags.writeable = False


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
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors (pymatgen convention).

        Returns:
            The minimum-image distance.
        """
        d0_base = frac1[0] - frac2[0]
        d1_base = frac1[1] - frac2[1]
        d2_base = frac1[2] - frac2[2]
        # Reduce to nearest integer so 27-image search covers all cases
        d0_base -= round(d0_base)
        d1_base -= round(d1_base)
        d2_base -= round(d2_base)

        min_dist_sq = np.inf
        for si in range(-1, 2):
            for sj in range(-1, 2):
                for sk in range(-1, 2):
                    d0 = d0_base + si
                    d1 = d1_base + sj
                    d2 = d2_base + sk
                    # Convert to Cartesian: d_frac @ lattice_matrix
                    cx = d0 * lattice_matrix[0, 0] + d1 * lattice_matrix[1, 0] + d2 * lattice_matrix[2, 0]
                    cy = d0 * lattice_matrix[0, 1] + d1 * lattice_matrix[1, 1] + d2 * lattice_matrix[2, 1]
                    cz = d0 * lattice_matrix[0, 2] + d1 * lattice_matrix[1, 2] + d2 * lattice_matrix[2, 2]
                    dist_sq = cx * cx + cy * cy + cz * cz
                    if dist_sq < min_dist_sq:
                        min_dist_sq = dist_sq
        return float(min_dist_sq ** 0.5)

    @numba.njit(cache=True, parallel=True)  # type: ignore[misc]
    def _all_mic_distances_numba(
        frac_coords1: np.ndarray,
        frac_coords2: np.ndarray,
        lattice_matrix: np.ndarray,
    ) -> np.ndarray:
        """JIT-compiled batch minimum-image distance matrix.

        Args:
            frac_coords1: Fractional coordinates, shape (N, 3).
            frac_coords2: Fractional coordinates, shape (M, 3).
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors (pymatgen convention).

        Returns:
            (N, M) array of minimum-image distances.
        """
        n = frac_coords1.shape[0]
        m = frac_coords2.shape[0]
        result = np.empty((n, m))
        for i in numba.prange(n):
            for j in range(m):
                d0_base = frac_coords1[i, 0] - frac_coords2[j, 0]
                d1_base = frac_coords1[i, 1] - frac_coords2[j, 1]
                d2_base = frac_coords1[i, 2] - frac_coords2[j, 2]
                d0_base -= round(d0_base)
                d1_base -= round(d1_base)
                d2_base -= round(d2_base)
                min_dist_sq = np.inf
                for si in range(-1, 2):
                    for sj in range(-1, 2):
                        for sk in range(-1, 2):
                            d0 = d0_base + si
                            d1 = d1_base + sj
                            d2 = d2_base + sk
                            cx = d0 * lattice_matrix[0, 0] + d1 * lattice_matrix[1, 0] + d2 * lattice_matrix[2, 0]
                            cy = d0 * lattice_matrix[0, 1] + d1 * lattice_matrix[1, 1] + d2 * lattice_matrix[2, 1]
                            cz = d0 * lattice_matrix[0, 2] + d1 * lattice_matrix[1, 2] + d2 * lattice_matrix[2, 2]
                            dist_sq = cx * cx + cy * cy + cz * cz
                            if dist_sq < min_dist_sq:
                                min_dist_sq = dist_sq
                result[i, j] = min_dist_sq ** 0.5
        return result


def mic_distance(
    frac1: np.ndarray,
    frac2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> float:
    """Minimum-image distance between two points in a periodic cell.

    Checks all 27 periodic images to find the true minimum distance.
    Uses numba JIT compilation when available for improved performance
    on repeated single-pair calls.

    Note:
        Behaviour is undefined for non-finite inputs (NaN, inf).

    Args:
        frac1: Fractional coordinates of point 1, shape (3,).
        frac2: Fractional coordinates of point 2, shape (3,).
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention: ``lattice.matrix``).

    Returns:
        The minimum-image distance in the same units as the lattice matrix.
    """
    if HAS_NUMBA:
        return float(_mic_distance_numba(frac1, frac2, lattice_matrix))
    d_frac = frac1 - frac2
    d_frac -= np.round(d_frac)
    # (27, 3) shifted difference vectors
    d_frac_all = d_frac + _SHIFTS_27
    # Convert to Cartesian and compute norms
    d_cart_all = d_frac_all @ lattice_matrix
    return float(np.min(np.linalg.norm(d_cart_all, axis=1)))  # type: ignore[arg-type]


def all_mic_distances(
    frac_coords1: np.ndarray,
    frac_coords2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Minimum-image distance matrix between two sets of points.

    Checks all 27 periodic images per pair to find true minimum
    distances, which is necessary for triclinic cells. Uses numba
    JIT compilation with parallel execution when available.

    Note:
        Behaviour is undefined for non-finite inputs (NaN, inf).

    Args:
        frac_coords1: Fractional coordinates, shape (N, 3).
        frac_coords2: Fractional coordinates, shape (M, 3).
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention: ``lattice.matrix``).

    Returns:
        (N, M) array of minimum-image distances in the same units as
            the lattice matrix.
    """
    if frac_coords1.shape[0] == 0 or frac_coords2.shape[0] == 0:
        return np.zeros((frac_coords1.shape[0], frac_coords2.shape[0]))
    if HAS_NUMBA:
        return np.asarray(_all_mic_distances_numba(frac_coords1, frac_coords2, lattice_matrix))
    # (N, 1, 3) - (1, M, 3) -> (N, M, 3) difference vectors
    d_frac = frac_coords1[:, np.newaxis, :] - frac_coords2[np.newaxis, :, :]
    d_frac -= np.round(d_frac)
    n, m = frac_coords1.shape[0], frac_coords2.shape[0]
    # Pre-allocate work buffers to avoid 54 temporary arrays across 27 iterations
    d_shifted = np.empty((n, m, 3))
    d_cart = np.empty((n, m, 3))
    dist_sq = np.empty((n, m))
    min_dist_sq = np.full((n, m), np.inf)
    for shift in _SHIFTS_27:
        np.add(d_frac, shift, out=d_shifted)
        np.matmul(d_shifted, lattice_matrix, out=d_cart)
        np.einsum("ijk,ijk->ij", d_cart, d_cart, out=dist_sq)
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)
    return np.asarray(np.sqrt(min_dist_sq))


def frac_to_cart(
    frac_coords: np.ndarray,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Convert fractional coordinates to Cartesian coordinates.

    Computes ``frac_coords @ lattice_matrix`` where lattice_matrix rows
    are the lattice vectors (pymatgen convention).

    Args:
        frac_coords: Fractional coordinates, shape (3,) or (N, 3).
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention: ``lattice.matrix``).

    Returns:
        Cartesian coordinates with the same shape as the input.
    """
    return np.asarray(frac_coords @ lattice_matrix)

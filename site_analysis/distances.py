"""Minimum-image distance and coordinate conversion functions for periodic systems.

Provides distance calculations and fractional-to-Cartesian coordinate
conversion operating on numpy arrays and a lattice matrix. Optional
numba acceleration for single-pair and paired distances.
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

# Below this many pairs, paired distances are computed on one thread,
# because dispatching work to numba's thread pool costs more than it saves.
# Measured on a 10-core machine; the break-even point grows with the number
# of threads.
_PARALLEL_MIN_PAIRS = 4096


if HAS_NUMBA:
    import numba

    @numba.njit(cache=True, inline="always")
    def _mic_distance_numba(
        frac1: np.ndarray,
        frac2: np.ndarray,
        lattice_matrix: np.ndarray,
    ) -> float:
        """JIT-compiled minimum-image distance over 27 periodic images.

        Inlined into the paired-distance kernels, so single-pair and paired
        distances use the same arithmetic.

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
        # Wrap to the image nearest in fractional coordinates. The 27 images
        # searched are those within one cell of it, which is exact when the
        # true distance is below the cell's smallest perpendicular width (#84).
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

    @numba.njit(cache=True)
    def _paired_mic_distances_serial(
        frac_coords1: np.ndarray,
        frac_coords2: np.ndarray,
        lattice_matrix: np.ndarray,
    ) -> np.ndarray:
        """JIT-compiled minimum-image distances between paired points, on one thread.

        Avoids the cost of dispatching work to numba's thread pool, which
        dominates for small batches.

        Args:
            frac_coords1: Fractional coordinates, shape (K, 3).
            frac_coords2: Fractional coordinates, shape (K, 3).
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors (pymatgen convention).

        Returns:
            (K,) array of minimum-image distances between
            ``frac_coords1[k]`` and ``frac_coords2[k]``.
        """
        n = frac_coords1.shape[0]
        result = np.empty(n)
        for p in range(n):
            result[p] = _mic_distance_numba(frac_coords1[p], frac_coords2[p], lattice_matrix)
        return result

    @numba.njit(cache=True, parallel=True)
    def _paired_mic_distances_parallel(
        frac_coords1: np.ndarray,
        frac_coords2: np.ndarray,
        lattice_matrix: np.ndarray,
    ) -> np.ndarray:
        """JIT-compiled minimum-image distances between paired points, in parallel.

        Args:
            frac_coords1: Fractional coordinates, shape (K, 3).
            frac_coords2: Fractional coordinates, shape (K, 3).
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors (pymatgen convention).

        Returns:
            (K,) array of minimum-image distances between
            ``frac_coords1[k]`` and ``frac_coords2[k]``.
        """
        n = frac_coords1.shape[0]
        result = np.empty(n)
        for p in numba.prange(n):
            result[p] = _mic_distance_numba(frac_coords1[p], frac_coords2[p], lattice_matrix)
        return result


def mic_distance(
    frac1: np.ndarray,
    frac2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> float:
    """Minimum-image distance between two points in a periodic cell.

    Checks the 27 periodic images nearest in fractional coordinates.
    This gives the true minimum distance whenever that distance is
    shorter than the cell's smallest perpendicular width (the smallest
    distance between opposite faces), and always in orthogonal cells. In
    thin or strongly skewed cells, such as a 1x10x1 hexagonal supercell,
    longer distances can be overestimated.
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
    return float(np.min(np.linalg.norm(d_cart_all, axis=1)))


def paired_mic_distances(
    frac_coords1: np.ndarray,
    frac_coords2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Minimum-image distances between corresponding pairs of points.

    Checks the 27 periodic images of each pair nearest in fractional
    coordinates. This gives the true minimum distance whenever that
    distance is shorter than the cell's smallest perpendicular width (the
    smallest distance between opposite faces), and always in orthogonal
    cells. In thin or strongly skewed cells, such as a 1x10x1 hexagonal
    supercell, longer distances can be overestimated.
    Uses numba JIT compilation when available, running large batches in
    parallel.

    Args:
        frac_coords1: Fractional coordinates, shape (K, 3).
        frac_coords2: Fractional coordinates, shape (K, 3).
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention: ``lattice.matrix``).

    Returns:
        (K,) array of minimum-image distances between
        ``frac_coords1[k]`` and ``frac_coords2[k]``, in the same units as
        the lattice matrix.

    Raises:
        ValueError: If the coordinate arrays do not both have shape
            (K, 3), or ``lattice_matrix`` does not have shape (3, 3), or
            any of the three is not finite.
    """
    frac_coords1 = np.ascontiguousarray(frac_coords1, dtype=np.float64)
    frac_coords2 = np.ascontiguousarray(frac_coords2, dtype=np.float64)
    lattice_matrix = np.ascontiguousarray(lattice_matrix, dtype=np.float64)
    if (frac_coords1.ndim != 2 or frac_coords1.shape[1] != 3
            or frac_coords1.shape != frac_coords2.shape):
        raise ValueError(
            f"frac_coords1 and frac_coords2 must both have shape (K, 3), "
            f"got {frac_coords1.shape} and {frac_coords2.shape}"
        )
    if lattice_matrix.shape != (3, 3):
        raise ValueError(
            f"lattice_matrix must have shape (3, 3), got {lattice_matrix.shape}"
        )
    for name, array in (("frac_coords1", frac_coords1),
                        ("frac_coords2", frac_coords2),
                        ("lattice_matrix", lattice_matrix)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
    return _paired_mic_distances(frac_coords1, frac_coords2, lattice_matrix)


def _paired_mic_distances(
    frac_coords1: np.ndarray,
    frac_coords2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Minimum-image distances between corresponding pairs of points, unchecked.

    As ``paired_mic_distances``, but without checking the inputs, for
    callers that have already validated them.

    Args:
        frac_coords1: Fractional coordinates, a contiguous float64 array
            of shape (K, 3).
        frac_coords2: Fractional coordinates, a contiguous float64 array
            of shape (K, 3).
        lattice_matrix: Contiguous float64 (3, 3) lattice matrix where
            rows are lattice vectors (pymatgen convention).

    Returns:
        (K,) array of minimum-image distances between
        ``frac_coords1[k]`` and ``frac_coords2[k]``.
    """
    if frac_coords1.shape[0] == 0:
        return np.zeros(0)
    if HAS_NUMBA:
        kernel = (_paired_mic_distances_parallel
                  if frac_coords1.shape[0] >= _PARALLEL_MIN_PAIRS
                  else _paired_mic_distances_serial)
        return np.asarray(kernel(frac_coords1, frac_coords2, lattice_matrix))
    d_frac = frac_coords1 - frac_coords2
    d_frac -= np.round(d_frac)
    min_dist_sq = np.full(d_frac.shape[0], np.inf)
    # Element-wise products and sums in the numba kernel's order, rather
    # than a matrix product: BLAS can round differently for batches of
    # different sizes, and a pair's distance must not depend on its batch.
    # This also gives the same distances as numba.
    for shift in _SHIFTS_27:
        d0 = d_frac[:, 0] + shift[0]
        d1 = d_frac[:, 1] + shift[1]
        d2 = d_frac[:, 2] + shift[2]
        cx = d0 * lattice_matrix[0, 0] + d1 * lattice_matrix[1, 0] + d2 * lattice_matrix[2, 0]
        cy = d0 * lattice_matrix[0, 1] + d1 * lattice_matrix[1, 1] + d2 * lattice_matrix[2, 1]
        cz = d0 * lattice_matrix[0, 2] + d1 * lattice_matrix[1, 2] + d2 * lattice_matrix[2, 2]
        np.minimum(min_dist_sq, cx * cx + cy * cy + cz * cz, out=min_dist_sq)
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

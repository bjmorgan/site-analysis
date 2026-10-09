"""Minimum-image distance and coordinate conversion functions for periodic systems.

Provides distance calculations and fractional-to-Cartesian coordinate
conversion operating on numpy arrays and a lattice matrix. Optional
numba acceleration for single-pair and paired distances.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

from site_analysis._compat import HAS_NUMBA


# Shift ranges are widened by this fraction of a cell, so that rounding
# cannot leave out an image that might be the nearest.
_SHIFT_MARGIN = 1e-9

# Below this many pairs, paired distances are computed on one thread,
# because dispatching work to numba's thread pool costs more than it saves.
# Measured on a 10-core machine; the break-even point grows with the number
# of threads.
_PARALLEL_MIN_PAIRS = 4096

_F = TypeVar("_F", bound=Callable[..., Any])
_T = TypeVar("_T", float, np.ndarray)


def _jitable(func: _F) -> _F:
    """Let numba-compiled code call ``func``, which stays a Python function.

    The numba kernels and the paths without numba then run the same
    arithmetic, so they give identical results.

    Args:
        func: A function that numba can compile.

    Returns:
        ``func`` itself.
    """
    if HAS_NUMBA:
        from numba.extending import register_jitable
        register_jitable(func)
    return func


@_jitable
def _squared_length(
    d0: _T,
    d1: _T,
    d2: _T,
    rows: Sequence[Sequence[float]] | np.ndarray,
) -> _T:
    """Squared Cartesian length of a fractional displacement.

    The products and sums are element-wise and in a fixed order, so that
    numba, Python floats and numpy arrays all round alike.

    Args:
        d0: First fractional component, a float or an array.
        d1: Second fractional component, of the same type as ``d0``.
        d2: Third fractional component, of the same type as ``d0``.
        rows: Lattice vectors as rows, indexed ``rows[i][j]``.

    Returns:
        The squared length, of the same type as the components.
    """
    cx = d0 * rows[0][0] + d1 * rows[1][0] + d2 * rows[2][0]
    cy = d0 * rows[0][1] + d1 * rows[1][1] + d2 * rows[2][1]
    cz = d0 * rows[0][2] + d1 * rows[1][2] + d2 * rows[2][2]
    return cx * cx + cy * cy + cz * cz


@_jitable
def _inverse_widths(
    rows: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[float, float, float]:
    """Reciprocals of a cell's three perpendicular widths.

    The width ``w_i`` is the distance between the two cell faces spanned
    by the other two lattice vectors, so ``1 / w_i = |a_j x a_k| / V``,
    the length of column ``i`` of the inverse lattice matrix. Any
    displacement of Cartesian length ``d`` has fractional components
    ``|f_i| <= d / w_i``.

    Args:
        rows: Lattice vectors as rows, indexed ``rows[i][j]``.

    Returns:
        ``(1 / w_0, 1 / w_1, 1 / w_2)``, or three NaNs if the cell volume
        is zero or not finite.
    """
    c0x = rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1]
    c0y = rows[1][2] * rows[2][0] - rows[1][0] * rows[2][2]
    c0z = rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0]
    c1x = rows[2][1] * rows[0][2] - rows[2][2] * rows[0][1]
    c1y = rows[2][2] * rows[0][0] - rows[2][0] * rows[0][2]
    c1z = rows[2][0] * rows[0][1] - rows[2][1] * rows[0][0]
    c2x = rows[0][1] * rows[1][2] - rows[0][2] * rows[1][1]
    c2y = rows[0][2] * rows[1][0] - rows[0][0] * rows[1][2]
    c2z = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
    volume = abs(rows[0][0] * c0x + rows[0][1] * c0y + rows[0][2] * c0z)
    if not (volume > 0.0 and volume < math.inf):
        return math.nan, math.nan, math.nan
    return (math.sqrt(c0x * c0x + c0y * c0y + c0z * c0z) / volume,
            math.sqrt(c1x * c1x + c1y * c1y + c1z * c1z) / volume,
            math.sqrt(c2x * c2x + c2y * c2y + c2z * c2z) / volume)


@_jitable
def _pair_distance(
    f0: float,
    f1: float,
    f2: float,
    rows: Sequence[Sequence[float]] | np.ndarray,
    widths: tuple[float, float, float],
) -> float:
    """Exact minimum-image distance for one fractional displacement.

    Starts from the image nearest in fractional coordinates,
    ``r = f - round(f)``, at Cartesian distance ``d``. Any image ``r + n``
    no further away has ``|r_i + n_i| <= d * widths[i]``, so only the
    integer shifts ``n`` in that box can be nearer. In most cells the box
    holds only ``n = 0``, and ``d`` is the answer.

    Args:
        f0: First component of the fractional displacement.
        f1: Second component of the fractional displacement.
        f2: Third component of the fractional displacement.
        rows: Lattice vectors as rows, indexed ``rows[i][j]``.
        widths: The cell's reciprocal perpendicular widths, from
            ``_inverse_widths``.

    Returns:
        The minimum-image distance.
    """
    r0 = f0 - round(f0)
    r1 = f1 - round(f1)
    r2 = f2 - round(f2)
    best = _squared_length(r0, r1, r2, rows)
    d = math.sqrt(best)
    lo0 = math.ceil(-d * widths[0] - r0 - _SHIFT_MARGIN)
    hi0 = math.floor(d * widths[0] - r0 + _SHIFT_MARGIN)
    lo1 = math.ceil(-d * widths[1] - r1 - _SHIFT_MARGIN)
    hi1 = math.floor(d * widths[1] - r1 + _SHIFT_MARGIN)
    lo2 = math.ceil(-d * widths[2] - r2 - _SHIFT_MARGIN)
    hi2 = math.floor(d * widths[2] - r2 + _SHIFT_MARGIN)
    if lo0 == 0 and hi0 == 0 and lo1 == 0 and hi1 == 0 and lo2 == 0 and hi2 == 0:
        return d
    for n0 in range(lo0, hi0 + 1):
        for n1 in range(lo1, hi1 + 1):
            for n2 in range(lo2, hi2 + 1):
                candidate = _squared_length(r0 + n0, r1 + n1, r2 + n2, rows)
                if candidate < best:
                    best = candidate
    return math.sqrt(best)


if HAS_NUMBA:
    import numba

    @numba.njit(cache=True, inline="always")
    def _lattice_rows(
        lattice_matrix: np.ndarray,
    ) -> tuple[tuple[float, float, float],
               tuple[float, float, float],
               tuple[float, float, float]]:
        """The lattice matrix as a tuple of rows, for ``_pair_distance``.

        Args:
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors (pymatgen convention).

        Returns:
            The three rows, each a tuple of three floats.
        """
        return ((lattice_matrix[0, 0], lattice_matrix[0, 1], lattice_matrix[0, 2]),
                (lattice_matrix[1, 0], lattice_matrix[1, 1], lattice_matrix[1, 2]),
                (lattice_matrix[2, 0], lattice_matrix[2, 1], lattice_matrix[2, 2]))

    @numba.njit(cache=True)
    def _mic_distance_numba(
        frac1: np.ndarray,
        frac2: np.ndarray,
        lattice_matrix: np.ndarray,
    ) -> float:
        """JIT-compiled exact minimum-image distance between two points.

        Args:
            frac1: Fractional coordinates of point 1, shape (3,).
            frac2: Fractional coordinates of point 2, shape (3,).
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors (pymatgen convention).

        Returns:
            The minimum-image distance.
        """
        rows = _lattice_rows(lattice_matrix)
        return _pair_distance(frac1[0] - frac2[0], frac1[1] - frac2[1],
                              frac1[2] - frac2[2], rows, _inverse_widths(rows))

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
        rows = _lattice_rows(lattice_matrix)
        widths = _inverse_widths(rows)
        n = frac_coords1.shape[0]
        result = np.empty(n)
        for p in range(n):
            result[p] = _pair_distance(
                frac_coords1[p, 0] - frac_coords2[p, 0],
                frac_coords1[p, 1] - frac_coords2[p, 1],
                frac_coords1[p, 2] - frac_coords2[p, 2],
                rows, widths)
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
        # Numba cannot compile a tuple of tuples in a parallel function, so
        # the lattice matrix itself stands in for the rows. Its elements
        # are the same floats, so the results equal the serial kernel's.
        widths = _inverse_widths(lattice_matrix)
        n = frac_coords1.shape[0]
        result = np.empty(n)
        for p in numba.prange(n):
            result[p] = _pair_distance(
                frac_coords1[p, 0] - frac_coords2[p, 0],
                frac_coords1[p, 1] - frac_coords2[p, 1],
                frac_coords1[p, 2] - frac_coords2[p, 2],
                lattice_matrix, widths)
        return result


def mic_distance(
    frac1: np.ndarray,
    frac2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> float:
    """Minimum-image distance between two points in a periodic cell.

    The shortest distance between any periodic images of the two points,
    exact for any cell. Uses numba JIT compilation when available for
    improved performance on repeated single-pair calls.

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
    return float(_paired_mic_distances_numpy(
        np.asarray(frac1, dtype=np.float64)[np.newaxis],
        np.asarray(frac2, dtype=np.float64)[np.newaxis],
        np.asarray(lattice_matrix, dtype=np.float64))[0])


def paired_mic_distances(
    frac_coords1: np.ndarray,
    frac_coords2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Minimum-image distances between corresponding pairs of points.

    For each pair, the shortest distance between any periodic images of
    the two points, exact for any cell. Uses numba JIT compilation when
    available, running large batches in parallel; the results are the
    same without numba.

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
    return _paired_mic_distances_numpy(frac_coords1, frac_coords2, lattice_matrix)


def _paired_mic_distances_numpy(
    frac_coords1: np.ndarray,
    frac_coords2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> np.ndarray:
    """Minimum-image distances between paired points, without numba.

    Args:
        frac_coords1: Fractional coordinates, a float64 array of shape
            (K, 3).
        frac_coords2: Fractional coordinates, a float64 array of shape
            (K, 3).
        lattice_matrix: Float64 (3, 3) lattice matrix where rows are
            lattice vectors (pymatgen convention).

    Returns:
        (K,) array of minimum-image distances between
        ``frac_coords1[k]`` and ``frac_coords2[k]``.
    """
    rows = lattice_matrix.tolist()
    return _distances_from_rounded_images(
        frac_coords1 - frac_coords2, rows, _inverse_widths(rows))


def _distances_from_rounded_images(
    frac_displacements: np.ndarray,
    rows: list[list[float]],
    widths: tuple[float, float, float],
) -> np.ndarray:
    """Exact minimum-image distances for many fractional displacements.

    ``_pair_distance`` vectorised: the rounded image of every
    displacement, then each shift in the combined box of the
    displacements that need more than one image, evaluated for just
    those whose own box contains it. The arithmetic follows
    ``_pair_distance`` step for step, so the results are identical.

    Args:
        frac_displacements: Fractional displacements, shape (K, 3).
        rows: Lattice vectors as rows, from ``lattice_matrix.tolist()``.
        widths: The cell's reciprocal perpendicular widths, from
            ``_inverse_widths``.

    Returns:
        (K,) array of minimum-image distances.
    """
    r = frac_displacements - np.round(frac_displacements)
    best = _squared_length(r[:, 0], r[:, 1], r[:, 2], rows)
    d = np.sqrt(best)[:, np.newaxis]
    inverse_widths = np.array(widths)
    lo = np.ceil(-d * inverse_widths - r - _SHIFT_MARGIN).astype(np.int64)
    hi = np.floor(d * inverse_widths - r + _SHIFT_MARGIN).astype(np.int64)
    search = np.flatnonzero(np.any((lo != 0) | (hi != 0), axis=1))
    if search.size:
        lo, hi = lo[search], hi[search]
        r_search, best_search = r[search], best[search]
        for n0 in range(lo[:, 0].min(), hi[:, 0].max() + 1):
            in_box_0 = (lo[:, 0] <= n0) & (n0 <= hi[:, 0])
            for n1 in range(lo[:, 1].min(), hi[:, 1].max() + 1):
                in_box_01 = in_box_0 & (lo[:, 1] <= n1) & (n1 <= hi[:, 1])
                for n2 in range(lo[:, 2].min(), hi[:, 2].max() + 1):
                    pairs = np.flatnonzero(
                        in_box_01 & (lo[:, 2] <= n2) & (n2 <= hi[:, 2]))
                    if pairs.size:
                        rp = r_search[pairs]
                        candidate = _squared_length(
                            rp[:, 0] + n0, rp[:, 1] + n1, rp[:, 2] + n2, rows)
                        best_search[pairs] = np.minimum(best_search[pairs], candidate)
        best[search] = best_search
    return np.asarray(np.sqrt(best))


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

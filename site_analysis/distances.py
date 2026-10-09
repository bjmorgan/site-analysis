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

# A cell whose volume is below this fraction of the product of its edge
# lengths counts as singular: its lattice vectors are (nearly) coplanar.
# Real cells are far above it (a rhombohedral cell with 10 degree angles is
# at 0.026). Cells only a little above it are valid but very flat, and the
# search for their nearest images can be slow.
_MIN_RELATIVE_VOLUME = 1e-8

# Floats this large or larger are whole numbers, so they are their own
# nearest integer. numba rounds to a 64-bit integer, which would overflow.
_WHOLE_NUMBER_LIMIT = 2.0 ** 52

# Boxes of more than this many shifts are searched by enumerating the ball
# of images that could be nearer (_search_ball) rather than the whole box
# (_search_box). The box is cheaper while it is small, as it is for every
# pair in a near-cubic cell; the ball skips most of the large boxes that
# long pairs get in elongated, thin or sheared cells, but builds its
# decomposition of the lattice for each pair. Profiled with numba over
# near-cubic, elongated, thin and sheared cells: at 64 no cell was more
# than about 5% slower than with the box alone, and elongated or sheared
# cells were up to 3.7 times faster; at 27 some thin cells were slower.
_MAX_BOX_SEARCH = 64

# Shifts of up to one cell along each axis. Without numba, pairs whose
# shift box fits within these get all 27 in one array operation, this many
# pairs at a time, which keeps the temporary arrays to a few megabytes.
_NEIGHBOUR_SHIFTS = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
    dtype=np.float64,
)
_NEIGHBOUR_SHIFTS.flags.writeable = False
_NEIGHBOUR_BLOCK = 8192

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
    numba, Python floats and numpy arrays all round alike. A matrix
    product would not: BLAS can round differently for batches of
    different sizes.

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
    ``|f_i| <= d / w_i``. The volume is computed from the unit vectors
    along the rows, so the test for a singular cell does not depend on
    the cell's scale.

    Args:
        rows: Lattice vectors as rows, indexed ``rows[i][j]``.

    Returns:
        ``(1 / w_0, 1 / w_1, 1 / w_2)``, or three NaNs if the cell is
        singular or nearly so (its volume is below ``_MIN_RELATIVE_VOLUME``
        times the product of its edge lengths) or not finite.
    """
    l0 = math.sqrt(rows[0][0] * rows[0][0] + rows[0][1] * rows[0][1] + rows[0][2] * rows[0][2])
    l1 = math.sqrt(rows[1][0] * rows[1][0] + rows[1][1] * rows[1][1] + rows[1][2] * rows[1][2])
    l2 = math.sqrt(rows[2][0] * rows[2][0] + rows[2][1] * rows[2][1] + rows[2][2] * rows[2][2])
    if not (0.0 < l0 < math.inf and 0.0 < l1 < math.inf and 0.0 < l2 < math.inf):
        return math.nan, math.nan, math.nan
    u00, u01, u02 = rows[0][0] / l0, rows[0][1] / l0, rows[0][2] / l0
    u10, u11, u12 = rows[1][0] / l1, rows[1][1] / l1, rows[1][2] / l1
    u20, u21, u22 = rows[2][0] / l2, rows[2][1] / l2, rows[2][2] / l2
    c0x = u11 * u22 - u12 * u21
    c0y = u12 * u20 - u10 * u22
    c0z = u10 * u21 - u11 * u20
    c1x = u21 * u02 - u22 * u01
    c1y = u22 * u00 - u20 * u02
    c1z = u20 * u01 - u21 * u00
    c2x = u01 * u12 - u02 * u11
    c2y = u02 * u10 - u00 * u12
    c2z = u00 * u11 - u01 * u10
    # The volume of the cell of unit vectors: the cell's volume divided by
    # the product of its edge lengths.
    relative_volume = abs(u00 * c0x + u01 * c0y + u02 * c0z)
    if not relative_volume > _MIN_RELATIVE_VOLUME:
        return math.nan, math.nan, math.nan
    return (math.sqrt(c0x * c0x + c0y * c0y + c0z * c0z) / (l0 * relative_volume),
            math.sqrt(c1x * c1x + c1y * c1y + c1z * c1z) / (l1 * relative_volume),
            math.sqrt(c2x * c2x + c2y * c2y + c2z * c2z) / (l2 * relative_volume))


@_jitable
def _inverse_widths_are_finite(inverse_widths: tuple[float, float, float]) -> bool:
    """Whether reciprocal widths are finite, which they are for a non-singular cell.

    Args:
        inverse_widths: Reciprocal perpendicular widths, from ``_inverse_widths``.

    Returns:
        ``True`` if all three are finite.
    """
    return inverse_widths[0] < math.inf and inverse_widths[1] < math.inf and inverse_widths[2] < math.inf


@_jitable
def _wrap(f: float) -> float:
    """A fractional coordinate minus its nearest whole number.

    Args:
        f: A finite fractional coordinate or displacement.

    Returns:
        ``f - round(f)``, in ``[-0.5, 0.5]``, with halves rounded to even.
    """
    if abs(f) >= _WHOLE_NUMBER_LIMIT:
        return 0.0
    return f - round(f)


@_jitable
def _shift_bounds(
    r0: float,
    r1: float,
    r2: float,
    d: float,
    inverse_widths: tuple[float, float, float],
) -> tuple[int, int, int, int, int, int]:
    """The box of integer shifts that holds every image within a distance.

    An image ``r + n`` at most ``d`` away has
    ``|r_i + n_i| <= d * inverse_widths[i]`` on each axis. The ranges are widened
    by ``_SHIFT_MARGIN`` so that rounding cannot leave out a shift.

    Args:
        r0: First component of the rounded fractional displacement.
        r1: Second component of the rounded fractional displacement.
        r2: Third component of the rounded fractional displacement.
        d: The distance to search within.
        inverse_widths: The cell's reciprocal perpendicular widths, from
            ``_inverse_widths``.

    Returns:
        ``(lo0, hi0, lo1, hi1, lo2, hi2)``, the inclusive range of shifts
        on each axis. Each range contains 0 when ``d`` is the length of
        ``r``.
    """
    return (math.ceil(-d * inverse_widths[0] - r0 - _SHIFT_MARGIN),
            math.floor(d * inverse_widths[0] - r0 + _SHIFT_MARGIN),
            math.ceil(-d * inverse_widths[1] - r1 - _SHIFT_MARGIN),
            math.floor(d * inverse_widths[1] - r1 + _SHIFT_MARGIN),
            math.ceil(-d * inverse_widths[2] - r2 - _SHIFT_MARGIN),
            math.floor(d * inverse_widths[2] - r2 + _SHIFT_MARGIN))


@_jitable
def _search_box(
    r0: float,
    r1: float,
    r2: float,
    best: float,
    rows: Sequence[Sequence[float]] | np.ndarray,
    inverse_widths: tuple[float, float, float],
) -> float:
    """Smallest squared length among the images that could be nearer.

    Evaluates every shift in the box from ``_shift_bounds``, which holds
    every image no further away than the rounded image.

    Args:
        r0: First component of the rounded fractional displacement.
        r1: Second component of the rounded fractional displacement.
        r2: Third component of the rounded fractional displacement.
        best: Squared length of the rounded image.
        rows: Lattice vectors as rows, indexed ``rows[i][j]``.
        inverse_widths: The cell's reciprocal perpendicular widths, from
            ``_inverse_widths``.

    Returns:
        The smallest squared length found, at most ``best``.
    """
    lo0, hi0, lo1, hi1, lo2, hi2 = _shift_bounds(r0, r1, r2, math.sqrt(best), inverse_widths)
    for n0 in range(lo0, hi0 + 1):
        for n1 in range(lo1, hi1 + 1):
            for n2 in range(lo2, hi2 + 1):
                candidate = _squared_length(r0 + n0, r1 + n1, r2 + n2, rows)
                if candidate < best:
                    best = candidate
    return best


@_jitable
def _enumeration_basis(
    rows: Sequence[Sequence[float]] | np.ndarray,
    inverse_widths: tuple[float, float, float],
) -> tuple[tuple[int, int, int, int, int, int],
           tuple[float, float, float, float, float, float, float, float, float]]:
    """The lattice decomposed for ``_search_ball``.

    The lattice vectors are taken in a cyclic order that ends with the
    axis of the widest perpendicular width, which the search enumerates
    first, and orthogonalised in that order (Gram-Schmidt). With ``y`` the
    fractional displacement in that order, the Cartesian displacement has
    components ``R22 * y2``, ``R11 * y1 + R12 * y2`` and
    ``R00 * y0 + R01 * y1 + R02 * y2`` along the third, second and first
    orthonormal directions.

    Args:
        rows: Lattice vectors as rows, indexed ``rows[i][j]``.
        inverse_widths: The cell's reciprocal perpendicular widths, from
            ``_inverse_widths``.

    Returns:
        ``(order, factors)``. ``order`` holds the axis taken at each of
        the three levels, then the level of each axis.
        ``factors`` is ``(1 / R00, R01 / R00, R02 / R00, 1 / R11,
        R12 / R11, 1 / R22, R11, R12, R22)``.
    """
    if inverse_widths[0] <= inverse_widths[1] and inverse_widths[0] <= inverse_widths[2]:
        p0, p1, p2, q0, q1, q2 = 1, 2, 0, 2, 0, 1
    elif inverse_widths[1] <= inverse_widths[2]:
        p0, p1, p2, q0, q1, q2 = 2, 0, 1, 1, 2, 0
    else:
        p0, p1, p2, q0, q1, q2 = 0, 1, 2, 0, 1, 2
    a, b, c = rows[p0], rows[p1], rows[p2]
    r00 = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    e00, e01, e02 = a[0] / r00, a[1] / r00, a[2] / r00
    r01 = b[0] * e00 + b[1] * e01 + b[2] * e02
    u10, u11, u12 = b[0] - r01 * e00, b[1] - r01 * e01, b[2] - r01 * e02
    r11 = math.sqrt(u10 * u10 + u11 * u11 + u12 * u12)
    e10, e11, e12 = u10 / r11, u11 / r11, u12 / r11
    r02 = c[0] * e00 + c[1] * e01 + c[2] * e02
    r12 = c[0] * e10 + c[1] * e11 + c[2] * e12
    u20 = c[0] - r02 * e00 - r12 * e10
    u21 = c[1] - r02 * e01 - r12 * e11
    u22 = c[2] - r02 * e02 - r12 * e12
    r22 = math.sqrt(u20 * u20 + u21 * u21 + u22 * u22)
    return ((p0, p1, p2, q0, q1, q2),
            (1.0 / r00, r01 / r00, r02 / r00, 1.0 / r11, r12 / r11, 1.0 / r22,
             r11, r12, r22))


@_jitable
def _search_ball(
    r0: float,
    r1: float,
    r2: float,
    best: float,
    rows: Sequence[Sequence[float]] | np.ndarray,
    inverse_widths: tuple[float, float, float],
) -> float:
    """Smallest squared length among the images that could be nearer.

    Enumerates the images within the ball of the current best distance
    (Fincke-Pohst), one axis at a time from the widest: each shift along
    an axis leaves less of the distance for the axes after it, so far
    fewer images are visited than in the box from ``_shift_bounds``, which
    bounds each axis on its own. Has the same arguments and result as
    ``_search_box``.

    Args:
        r0: First component of the rounded fractional displacement.
        r1: Second component of the rounded fractional displacement.
        r2: Third component of the rounded fractional displacement.
        best: Squared length of the rounded image.
        rows: Lattice vectors as rows, indexed ``rows[i][j]``.
        inverse_widths: The cell's reciprocal perpendicular widths, from
            ``_inverse_widths``.

    Returns:
        The smallest squared length found, at most ``best``.
    """
    order, factors = _enumeration_basis(rows, inverse_widths)
    i_r00, r01_r00, r02_r00, i_r11, r12_r11, i_r22, r11, r12, r22 = factors
    r = (r0, r1, r2)
    y0c, y1c, y2c = r[order[0]], r[order[1]], r[order[2]]
    # Images up to a relative margin beyond the current best are visited,
    # so that rounding in these bounds does not leave out a nearer image in
    # any but nearly singular cells.
    slack = 1.0 + _SHIFT_MARGIN
    d = math.sqrt(best * slack)
    for n2 in range(math.ceil(-d * i_r22 - y2c - _SHIFT_MARGIN),
                    math.floor(d * i_r22 - y2c + _SHIFT_MARGIN) + 1):
        y2 = y2c + n2
        t2 = r22 * y2
        rest2 = best * slack - t2 * t2
        if rest2 < 0.0:
            continue
        s1 = math.sqrt(rest2) * i_r11
        c1 = r12_r11 * y2
        for n1 in range(math.ceil(-s1 - c1 - y1c - _SHIFT_MARGIN),
                        math.floor(s1 - c1 - y1c + _SHIFT_MARGIN) + 1):
            y1 = y1c + n1
            t1 = r11 * y1 + r12 * y2
            rest1 = rest2 - t1 * t1
            if rest1 < 0.0:
                continue
            s0 = math.sqrt(rest1) * i_r00
            c0 = r01_r00 * y1 + r02_r00 * y2
            for n0 in range(math.ceil(-s0 - c0 - y0c - _SHIFT_MARGIN),
                            math.floor(s0 - c0 - y0c + _SHIFT_MARGIN) + 1):
                n = (n0, n1, n2)
                # Scored with _squared_length in the original axis order, not
                # from the remaining distance, so that results are identical
                # to _search_box's.
                candidate = _squared_length(
                    r0 + n[order[3]], r1 + n[order[4]], r2 + n[order[5]], rows)
                if candidate < best:
                    best = candidate
    return best


@_jitable
def _pair_distance(
    f0: float,
    f1: float,
    f2: float,
    rows: Sequence[Sequence[float]] | np.ndarray,
    inverse_widths: tuple[float, float, float],
) -> float:
    """Exact minimum-image distance for one fractional displacement.

    Starts from the image nearest in fractional coordinates,
    ``r = f - round(f)``, at Cartesian distance ``d``. Only the integer
    shifts ``n`` in the box from ``_shift_bounds`` can give a nearer image
    ``r + n``. For pairs closer than about half the cell's smallest
    perpendicular width the box holds only ``n = 0``, and ``d`` is the
    answer. Otherwise ``_search_box`` searches the box, or, if it holds
    more than ``_MAX_BOX_SEARCH`` shifts, ``_search_ball`` searches the
    ball of images that could be nearer.

    Args:
        f0: First component of the fractional displacement.
        f1: Second component of the fractional displacement.
        f2: Third component of the fractional displacement.
        rows: Lattice vectors as rows, indexed ``rows[i][j]``.
        inverse_widths: The cell's reciprocal perpendicular widths, from
            ``_inverse_widths``.

    Returns:
        The minimum-image distance.
    """
    # Non-finite input or a singular lattice: there is no distance to find,
    # and rounding NaN would raise in Python.
    if not (abs(f0) < math.inf and abs(f1) < math.inf and abs(f2) < math.inf
            and _inverse_widths_are_finite(inverse_widths)):
        return math.nan
    r0 = _wrap(f0)
    r1 = _wrap(f1)
    r2 = _wrap(f2)
    best = _squared_length(r0, r1, r2, rows)
    d = math.sqrt(best)
    lo0, hi0, lo1, hi1, lo2, hi2 = _shift_bounds(r0, r1, r2, d, inverse_widths)
    if lo0 == 0 and hi0 == 0 and lo1 == 0 and hi1 == 0 and lo2 == 0 and hi2 == 0:
        return d
    # Counted in floats: in a nearly singular cell the count can exceed a
    # 64-bit integer, which numba would wrap round to a negative number.
    if (hi0 - lo0 + 1.0) * (hi1 - lo1 + 1.0) * (hi2 - lo2 + 1.0) <= _MAX_BOX_SEARCH:
        return math.sqrt(_search_box(r0, r1, r2, best, rows, inverse_widths))
    return math.sqrt(_search_ball(r0, r1, r2, best, rows, inverse_widths))


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
        inverse_widths = _inverse_widths(rows)
        n = frac_coords1.shape[0]
        result = np.empty(n)
        for p in range(n):
            result[p] = _pair_distance(
                frac_coords1[p, 0] - frac_coords2[p, 0],
                frac_coords1[p, 1] - frac_coords2[p, 1],
                frac_coords1[p, 2] - frac_coords2[p, 2],
                rows, inverse_widths)
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
        inverse_widths = _inverse_widths(lattice_matrix)
        n = frac_coords1.shape[0]
        result = np.empty(n)
        for p in numba.prange(n):
            result[p] = _pair_distance(
                frac_coords1[p, 0] - frac_coords2[p, 0],
                frac_coords1[p, 1] - frac_coords2[p, 1],
                frac_coords1[p, 2] - frac_coords2[p, 2],
                lattice_matrix, inverse_widths)
        return result


# The rows and reciprocal widths of the last lattice that ``mic_distance``
# was given without numba, keyed by the lattice's bytes. Computing them
# takes about as long as a distance, and callers such as spherical sites
# ask for many distances in one lattice. One tuple, so that a reader never
# pairs one lattice's key with another lattice's values.
_last_lattice: tuple[bytes, list[list[float]], tuple[float, float, float]] = (
    b"", [], (math.nan, math.nan, math.nan))


def _rows_and_inverse_widths(
    lattice_matrix: np.ndarray,
) -> tuple[list[list[float]], tuple[float, float, float]]:
    """Lattice rows as Python floats, and reciprocal widths, kept for reuse.

    Args:
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention).

    Returns:
        ``(rows, inverse_widths)`` for ``_pair_distance``.
    """
    global _last_lattice
    stored = _last_lattice
    key = lattice_matrix.tobytes()
    if key != stored[0]:
        rows = lattice_matrix.tolist()
        stored = (key, rows, _inverse_widths(rows))
        _last_lattice = stored
    return stored[1], stored[2]


def mic_distance(
    frac1: np.ndarray,
    frac2: np.ndarray,
    lattice_matrix: np.ndarray,
) -> float:
    """Minimum-image distance between two points in a periodic cell.

    The shortest distance between any periodic images of the two points,
    exact for any cell. Uses numba JIT compilation when available for
    improved performance on repeated single-pair calls; for float64
    input, the result is the same without numba.

    Note:
        The distance is undefined for non-finite inputs (NaN, inf) or a
        lattice matrix that is singular or nearly so, and NaN is returned.

    Args:
        frac1: Fractional coordinates of point 1, a float64 array of
            shape (3,).
        frac2: Fractional coordinates of point 2, a float64 array of
            shape (3,).
        lattice_matrix: (3, 3) lattice matrix where rows are lattice
            vectors (pymatgen convention: ``lattice.matrix``).

    Returns:
        The minimum-image distance in the same units as the lattice matrix.
    """
    if HAS_NUMBA:
        return float(_mic_distance_numba(frac1, frac2, lattice_matrix))
    rows, inverse_widths = _rows_and_inverse_widths(lattice_matrix)
    return _pair_distance(float(frac1[0]) - float(frac2[0]),
                          float(frac1[1]) - float(frac2[1]),
                          float(frac1[2]) - float(frac2[2]),
                          rows, inverse_widths)


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
            any of the three is not finite, or the lattice matrix is
            singular or nearly so.
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
    if not _inverse_widths_are_finite(_inverse_widths(lattice_matrix.tolist())):
        raise ValueError("lattice_matrix must be non-singular")
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
        ``frac_coords1[k]`` and ``frac_coords2[k]``. NaN for a pair whose
        coordinates are not finite, and for every pair if the lattice
        matrix is singular or nearly so.
    """
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
        ``frac_coords1[k]`` and ``frac_coords2[k]``. NaN for a pair whose
        coordinates are not finite, and for every pair if the lattice
        matrix is singular or nearly so.
    """
    rows = lattice_matrix.tolist()
    inverse_widths = _inverse_widths(rows)
    frac_displacements = frac_coords1 - frac_coords2
    distances = np.full(frac_displacements.shape[0], np.nan)
    if not _inverse_widths_are_finite(inverse_widths):
        return distances
    # Non-finite pairs are left out: their shift bounds would be NaN, and
    # converting NaN to an integer gives a different, meaningless number on
    # each platform.
    finite = np.isfinite(frac_displacements).all(axis=1)
    if finite.all():
        return _distances_from_rounded_images(frac_displacements, rows, inverse_widths)
    distances[finite] = _distances_from_rounded_images(
        frac_displacements[finite], rows, inverse_widths)
    return distances


def _distances_from_rounded_images(
    frac_displacements: np.ndarray,
    rows: list[list[float]],
    inverse_widths: tuple[float, float, float],
) -> np.ndarray:
    """Exact minimum-image distances for many fractional displacements.

    ``_pair_distance`` vectorised: the rounded image of every
    displacement, then, for those whose box holds more than one image,
    every shift of up to one cell each way at once if their box fits
    within that; ``_search_ball`` for each displacement whose box holds
    more than ``_MAX_BOX_SEARCH`` shifts, as ``_pair_distance`` does; and
    for the rest, each shift in their combined box, evaluated for just
    those whose own box contains it. Each image's arithmetic follows
    ``_pair_distance`` step for step, and any extra images evaluated are
    further than the rounded image, so the results are identical.

    Args:
        frac_displacements: Fractional displacements, shape (K, 3).
        rows: Lattice vectors as rows, from ``lattice_matrix.tolist()``.
        inverse_widths: The cell's reciprocal perpendicular widths, from
            ``_inverse_widths``.

    Returns:
        (K,) array of minimum-image distances.
    """
    r = frac_displacements - np.round(frac_displacements)
    best = _squared_length(r[:, 0], r[:, 1], r[:, 2], rows)
    d = np.sqrt(best)[:, np.newaxis]
    per_axis = np.array(inverse_widths)
    lo = np.ceil(-d * per_axis - r - _SHIFT_MARGIN)
    hi = np.floor(d * per_axis - r + _SHIFT_MARGIN)
    search = np.flatnonzero(np.any((lo != 0) | (hi != 0), axis=1))
    if search.size == 0:
        return np.asarray(np.sqrt(best))
    within_one_cell = np.all((lo[search] >= -1) & (hi[search] <= 1), axis=1)
    # Boxes within one cell each way, which include every long pair in a
    # near-cubic cell: all 27 such shifts at once, a block of pairs at a
    # time. A shift outside a pair's own box gives an image further away
    # than its rounded image, so it cannot change the result.
    near = search[within_one_cell]
    for start in range(0, near.size, _NEIGHBOUR_BLOCK):
        pairs = near[start:start + _NEIGHBOUR_BLOCK]
        rp = r[pairs]
        candidate = _squared_length(
            rp[:, 0, np.newaxis] + _NEIGHBOUR_SHIFTS[:, 0],
            rp[:, 1, np.newaxis] + _NEIGHBOUR_SHIFTS[:, 1],
            rp[:, 2, np.newaxis] + _NEIGHBOUR_SHIFTS[:, 2], rows)
        best[pairs] = np.minimum(best[pairs], candidate.min(axis=1))
    # Larger boxes, in thin, sheared or elongated cells, counted in floats
    # as they can exceed a 64-bit integer. Those of more than
    # _MAX_BOX_SEARCH shifts: the ball search, pair by pair, as with numba.
    rest = search[~within_one_cell]
    box_size = np.prod(hi[rest] - lo[rest] + 1.0, axis=1)
    for p in rest[box_size > _MAX_BOX_SEARCH]:
        best[p] = _search_ball(float(r[p, 0]), float(r[p, 1]), float(r[p, 2]),
                               float(best[p]), rows, inverse_widths)
    # The others: each shift in their combined box, for just the pairs whose
    # own box contains it.
    far = rest[box_size <= _MAX_BOX_SEARCH]
    if far.size:
        lo, hi = lo[far].astype(np.int64), hi[far].astype(np.int64)
        r_search, best_search = r[far], best[far]
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
        best[far] = best_search
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

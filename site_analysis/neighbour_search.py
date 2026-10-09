"""Periodic neighbour search with a KD-tree.

Provides ``PeriodicNeighbourIndex``, which finds the indexed points
within a cutoff of each query point, or the nearest indexed point, in a
periodic cell, including non-orthogonal cells. Distances are exact
minimum-image distances, those of ``paired_mic_distances``, and results
equal those of computing that distance from every query point to every
indexed point, with memory that scales with the number of points and
pairs found.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import cast

import numpy as np
from scipy.spatial import cKDTree

from site_analysis.distances import (
    _inverse_widths,
    _inverse_widths_are_finite,
    _paired_mic_distances,
)

# The rows of the normalised lattice are unit vectors, so a smallest
# singular value below this means the lattice vectors are (almost)
# coplanar.
_MIN_SINGULAR_VALUE = 1e-8

# Candidate searches are widened slightly so that rounding does not drop a
# neighbour. Exact distances then remove any extra candidates.
_RELATIVE_TOLERANCE = 1e-9
_ABSOLUTE_TOLERANCE = 1e-12

# Exact distances are computed from unwrapped coordinates, so their
# rounding error grows with the coordinates' distance from the cell.
# Searches are also widened by this multiple of that error bound.
_ROUNDING_SAFETY = 8.0


def _as_coords(coords: np.ndarray, name: str) -> np.ndarray:
    """Return coordinates as a float64 array of shape (N, 3).

    Args:
        coords: Coordinates to check.
        name: Argument name, for the error message.

    Returns:
        The coordinates as a contiguous float64 array.

    Raises:
        ValueError: If ``coords`` does not have shape (N, 3) or is not
            finite.
    """
    array = np.ascontiguousarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


class PeriodicNeighbourIndex:
    """KD-tree neighbour search for points in a periodic cell.

    The tree holds fractional coordinates scaled by the lattice lengths,
    in which the cell is an axis-aligned box. Where the cell angles are
    not 90 degrees, separations in these coordinates differ from Cartesian
    separations, but a Cartesian separation is never shorter than
    ``sigma_min`` times the scaled separation, where ``sigma_min`` is the
    smallest singular value of the lattice matrix with each row scaled to
    unit length. A tree search with radius ``r / sigma_min`` therefore
    finds every pair within Cartesian distance ``r``. The exact
    minimum-image distance of each candidate pair is then computed by
    ``paired_mic_distances``, so results match that function.

    An index is fixed to the lattice it was built with. Build a new index
    if the lattice changes.
    """

    def __init__(self,
            frac_coords: np.ndarray,
            lattice_matrix: np.ndarray) -> None:
        """Create a PeriodicNeighbourIndex.

        Args:
            frac_coords: Fractional coordinates of the points to index,
                shape (N, 3).
            lattice_matrix: (3, 3) lattice matrix where rows are lattice
                vectors (pymatgen convention: ``lattice.matrix``).

        Raises:
            ValueError: If ``frac_coords`` does not have shape (N, 3) or is
                not finite, or ``lattice_matrix`` does not have shape
                (3, 3), is not finite, or is singular or nearly so.
        """
        self._frac_coords = _as_coords(frac_coords, "frac_coords").copy()
        self._lattice_matrix = np.array(lattice_matrix, dtype=np.float64, order="C")
        if self._lattice_matrix.shape != (3, 3):
            raise ValueError(
                f"lattice_matrix must have shape (3, 3), got {self._lattice_matrix.shape}"
            )
        if not np.isfinite(self._lattice_matrix).all():
            raise ValueError("lattice_matrix must be finite")
        self._lengths = np.linalg.norm(self._lattice_matrix, axis=1)
        if np.any(self._lengths == 0.0):
            raise ValueError("lattice_matrix must be non-singular, but has a zero-length row")
        unit_rows = self._lattice_matrix / self._lengths[:, np.newaxis]
        self._sigma_min = float(np.linalg.svd(unit_rows, compute_uv=False).min())
        # The second test is the one the distance functions apply, so that no
        # lattice accepted here gives them undefined distances.
        if (self._sigma_min < _MIN_SINGULAR_VALUE
                or not _inverse_widths_are_finite(_inverse_widths(self._lattice_matrix.tolist()))):
            raise ValueError("lattice_matrix must be non-singular, but its rows are (nearly) coplanar")
        self._max_abs_coord = float(np.abs(self._frac_coords).max(initial=0.0))
        self._tree = cKDTree(self._scaled(self._frac_coords), boxsize=self._lengths)

    def __len__(self) -> int:
        """Return the number of indexed points."""
        return int(self._frac_coords.shape[0])

    def _scaled(self, frac_coords: np.ndarray) -> np.ndarray:
        """Map fractional coordinates into the tree's box.

        Args:
            frac_coords: Fractional coordinates, shape (N, 3).

        Returns:
            Coordinates wrapped into [0, 1) and scaled by the lattice
            lengths.
        """
        scaled = np.mod(frac_coords, 1.0) * self._lengths
        # np.mod returns exactly 1.0 for tiny negative inputs, and cKDTree
        # rejects points on the far edge of the box.
        return np.where(scaled >= self._lengths, 0.0, scaled)

    def _search_radius(self,
            distance: float | np.ndarray,
            query_frac: np.ndarray) -> np.ndarray:
        """Return the tree search radius that covers a Cartesian distance.

        The radius is widened slightly so that rounding does not drop a
        neighbour. Rounding errors in the exact distances grow with the
        magnitude of the fractional coordinates, and so does the widening.

        Args:
            distance: Cartesian distance, or one distance per query point.
            query_frac: Fractional coordinates of the query points,
                shape (M, 3).

        Returns:
            The radius in scaled coordinates for each query point,
            shape (M,).
        """
        coord_scale = np.abs(query_frac).max(axis=1) + self._max_abs_coord + 1.0
        rounding = (_ROUNDING_SAFETY * np.finfo(np.float64).eps
                    * coord_scale * self._lengths.max())
        return ((distance + rounding) / self._sigma_min * (1.0 + _RELATIVE_TOLERANCE)
                + _ABSOLUTE_TOLERANCE)

    def _candidates(self,
            query_frac: np.ndarray,
            radius: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find candidate pairs in the tree and compute their exact distances.

        Args:
            query_frac: Fractional coordinates of the query points,
                shape (M, 3).
            radius: Tree search radius for each query point, shape (M,).

        Returns:
            Tuple of ``(query_idx, point_idx, distances)``, one entry per
            candidate pair, in no particular order.
        """
        if query_frac.shape[0] == 0 or len(self) == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0)
        # For an (M, 3) query, query_ball_point returns M lists of indices.
        neighbour_lists = cast(Sequence[list[int]], self._tree.query_ball_point(
            self._scaled(query_frac), radius, return_sorted=False))
        counts = np.fromiter((len(n) for n in neighbour_lists),
                             dtype=np.intp, count=len(neighbour_lists))
        point_idx = np.fromiter(itertools.chain.from_iterable(neighbour_lists),
                                dtype=np.intp, count=int(counts.sum()))
        query_idx = np.repeat(np.arange(len(neighbour_lists), dtype=np.intp), counts)
        distances = _paired_mic_distances(
            query_frac[query_idx], self._frac_coords[point_idx], self._lattice_matrix)
        return query_idx, point_idx, distances

    def query_within(self,
            query_frac: np.ndarray,
            cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the indexed points within a cutoff of each query point.

        Args:
            query_frac: Fractional coordinates of the query points,
                shape (M, 3).
            cutoff: Distance cutoff, in the units of the lattice matrix;
                ``np.inf`` returns every pair.
                Pairs with a minimum-image distance ``<= cutoff``,
                including those exactly at the cutoff, are returned.

        Returns:
            Tuple of ``(query_idx, point_idx, distances)``, one entry per
            pair, sorted by query index, then distance, then point index.
            Indices are ``np.intp`` arrays and distances a ``float64``
            array.

        Raises:
            ValueError: If ``query_frac`` does not have shape (M, 3) or is
                not finite, or ``cutoff`` is negative or NaN.
        """
        query_frac = _as_coords(query_frac, "query_frac")
        if not cutoff >= 0:
            raise ValueError(f"cutoff must be non-negative, got {cutoff}")
        query_idx, point_idx, distances = self._candidates(
            query_frac, self._search_radius(cutoff, query_frac))
        within = distances <= cutoff
        query_idx, point_idx, distances = query_idx[within], point_idx[within], distances[within]
        order = np.lexsort((point_idx, distances, query_idx))
        return query_idx[order], point_idx[order], distances[order]

    def query_nearest(self,
            query_frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find the nearest indexed point to each query point.

        Ties are broken by the lowest point index.

        Args:
            query_frac: Fractional coordinates of the query points,
                shape (M, 3).

        Returns:
            Tuple of ``(point_idx, distances)``, each of length M. Indices
            are an ``np.intp`` array and distances a ``float64`` array.

        Raises:
            ValueError: If ``query_frac`` does not have shape (M, 3) or is
                not finite, or the index is empty.
            RuntimeError: If some query point has no candidate. The tree's
                nearest point is always a candidate, so this would mean a
                bug in the search.
        """
        query_frac = _as_coords(query_frac, "query_frac")
        if len(self) == 0:
            raise ValueError("Cannot find nearest neighbours in an empty index")
        if query_frac.shape[0] == 0:
            return np.empty(0, dtype=np.intp), np.empty(0)
        # The exact distance to the tree's nearest point is an upper bound
        # on the nearest minimum-image distance.
        _, first = self._tree.query(self._scaled(query_frac), k=1)
        upper = _paired_mic_distances(
            query_frac, self._frac_coords[first], self._lattice_matrix)
        query_idx, point_idx, distances = self._candidates(
            query_frac, self._search_radius(upper, query_frac))
        order = np.lexsort((point_idx, distances, query_idx))
        _, first_per_query = np.unique(query_idx[order], return_index=True)
        if len(first_per_query) != query_frac.shape[0]:
            raise RuntimeError("query_nearest found no candidate for some query points")
        nearest = order[first_per_query]
        return point_idx[nearest], distances[nearest]

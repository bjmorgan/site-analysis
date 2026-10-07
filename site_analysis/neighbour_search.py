"""Periodic neighbour search with a KD-tree.

Provides ``PeriodicNeighbourIndex``, which finds the indexed points
within a cutoff of each query point, or the nearest indexed point, in a
periodic cell of any shape. Results equal those of computing the
minimum-image distance from every query point to every indexed point,
with memory that scales with the number of points and pairs found.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import cast

import numpy as np
from scipy.spatial import cKDTree

from site_analysis.distances import paired_mic_distances

# The rows of the normalised lattice are unit vectors, so a smallest
# singular value below this means the lattice vectors are (almost)
# coplanar.
_MIN_SINGULAR_VALUE = 1e-8

# Candidate searches are widened slightly so that rounding cannot drop a
# neighbour. Exact distances then remove any extra candidates.
_RELATIVE_TOLERANCE = 1e-9
_ABSOLUTE_TOLERANCE = 1e-12


def _as_coords(coords: np.ndarray, name: str) -> np.ndarray:
    """Return coordinates as a float64 array of shape (N, 3).

    Args:
        coords: Coordinates to check.
        name: Argument name, for the error message.

    Returns:
        The coordinates as a contiguous float64 array.

    Raises:
        ValueError: If ``coords`` does not have shape (N, 3).
    """
    array = np.ascontiguousarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
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
    finds every pair within Cartesian distance ``r``. The distance of each
    candidate pair is then computed over 27 periodic images by
    ``paired_mic_distances``.
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
            ValueError: If ``frac_coords`` does not have shape (N, 3), or
                ``lattice_matrix`` does not have shape (3, 3) or is
                singular.
        """
        self._frac_coords = _as_coords(frac_coords, "frac_coords").copy()
        self._lattice_matrix = np.array(lattice_matrix, dtype=np.float64, order="C")
        if self._lattice_matrix.shape != (3, 3):
            raise ValueError(
                f"lattice_matrix must have shape (3, 3), got {self._lattice_matrix.shape}"
            )
        self._lengths = np.linalg.norm(self._lattice_matrix, axis=1)
        if np.any(self._lengths == 0.0):
            raise ValueError("lattice_matrix must be non-singular, but has a zero-length row")
        unit_rows = self._lattice_matrix / self._lengths[:, np.newaxis]
        self._sigma_min = float(np.linalg.svd(unit_rows, compute_uv=False).min())
        if self._sigma_min < _MIN_SINGULAR_VALUE:
            raise ValueError("lattice_matrix must be non-singular, but its rows are coplanar")
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

    def _search_radius(self, distance: float | np.ndarray) -> float | np.ndarray:
        """Return the tree search radius that covers a Cartesian distance.

        Args:
            distance: Cartesian distance, or one distance per query point.

        Returns:
            The radius in scaled coordinates, widened slightly for
            rounding.
        """
        return distance / self._sigma_min * (1.0 + _RELATIVE_TOLERANCE) + _ABSOLUTE_TOLERANCE

    def _candidates(self,
            query_frac: np.ndarray,
            radius: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find candidate pairs in the tree and compute their exact distances.

        Args:
            query_frac: Fractional coordinates of the query points,
                shape (M, 3).
            radius: Tree search radius, or one radius per query point.

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
        distances = paired_mic_distances(
            query_frac[query_idx], self._frac_coords[point_idx], self._lattice_matrix)
        return query_idx, point_idx, distances

    def query_within(self,
            query_frac: np.ndarray,
            cutoff: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find the indexed points within a cutoff of each query point.

        Args:
            query_frac: Fractional coordinates of the query points,
                shape (M, 3).
            cutoff: Distance cutoff, in the units of the lattice matrix.
                Pairs with a minimum-image distance ``<= cutoff``,
                including those exactly at the cutoff, are returned.

        Returns:
            Tuple of ``(query_idx, point_idx, distances)``, one entry per
            pair, sorted by query index, then distance, then point index.
            Indices are ``np.intp`` arrays and distances a ``float64``
            array.

        Raises:
            ValueError: If ``query_frac`` does not have shape (M, 3), or
                ``cutoff`` is negative or NaN.
        """
        query_frac = _as_coords(query_frac, "query_frac")
        if not cutoff >= 0:
            raise ValueError(f"cutoff must be non-negative, got {cutoff}")
        query_idx, point_idx, distances = self._candidates(
            query_frac, self._search_radius(cutoff))
        within = distances <= cutoff
        query_idx, point_idx, distances = query_idx[within], point_idx[within], distances[within]
        order = np.lexsort((point_idx, distances, query_idx))
        return query_idx[order], point_idx[order], distances[order]
